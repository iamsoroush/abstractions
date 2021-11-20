import pathlib
import argparse
import sys
from pydoc import locate

import mlflow
import pandas as pd
import tensorflow.keras as tfk

from abstractions import *
from abstractions.utils import load_config_file, check_for_config_file, setup_mlflow, get_logger

MLFLOW_TRACKING_URI = pathlib.Path('/home').joinpath('vafaeisa').joinpath('mlruns')
SCRATCH_PATH = pathlib.Path('/home').joinpath('vafaeisa').joinpath('scratch')
EVAL_REPORTS_DIR = SCRATCH_PATH.joinpath('eval_reports')


class Orchestrator:

    def __init__(self, run_dir: pathlib.Path, data_dir: pathlib.Path):
        self.run_dir = run_dir
        self.run_name = self.run_dir.name
        self.data_dir = data_dir

        self.logger = get_logger('orchestrator')

        config_path = check_for_config_file(run_dir)
        self.config = load_config_file(config_path.absolute())
        self.logger.info(f'config file loaded: {config_path}')

        # load params
        self.project_name = self.config.project_name
        self.src_code_path = self.config.src_code_path
        self.do_train_augmentation = self.config.do_train_augmentation
        self.do_validation_augmentation = self.config.do_validation_augmentation

        self.eval_report_dir = EVAL_REPORTS_DIR.joinpath(self.project_name).joinpath(self.run_name)
        self.eval_report_dir.mkdir(parents=True, exist_ok=True)

        sys.path.append(self.src_code_path)
        self.logger.info(f'{self.src_code_path} has been added to system paths.')

        # Instantiating components
        self.data_loader = self._create_data_loader()
        self.augmentor = self._create_augmentor()
        self.preprocessor = self._create_preprocessor()
        self.model_builder = self._create_model_builder()
        self.trainer = self._create_trainer()
        self.evaluator = self._create_evaluator()

        self.mlflow_tracking_uri = MLFLOW_TRACKING_URI

    def run(self):
        # data generators
        train_data_gen, train_n = self.data_loader.create_training_generator()
        validation_data_gen, validation_n = self.data_loader.create_validation_generator()
        self.logger.info('data-generators has been created')

        # augmentation
        if self.augmentor is not None:
            if self.do_train_augmentation:
                train_data_gen = self.augmentor.add_augmentation(train_data_gen)
                self.logger.info('added training augmentations')
            if self.do_validation_augmentation:
                validation_data_gen = self.augmentor.add_augmentation(validation_data_gen)
                self.logger.info('added validation augmentations')

        # preprocessing
        train_data_gen, n_iter_train = self.preprocessor.add_preprocess(train_data_gen, train_n)
        validation_data_gen, n_iter_val = self.preprocessor.add_preprocess(validation_data_gen, validation_n)
        self.logger.info('added preprocessing to data-generators')

        # training
        mlflow_active_run = setup_mlflow(mlflow_tracking_uri=str(self.mlflow_tracking_uri),
                                         mlflow_experiment_name=self.project_name,
                                         base_dir=self.run_dir)

        self.logger.info('training started ...')
        self.trainer.train(model_builder=self.model_builder,
                           active_run=mlflow_active_run,
                           train_data_gen=train_data_gen,
                           n_iter_train=n_iter_train,
                           val_data_gen=validation_data_gen,
                           n_iter_val=n_iter_val)

        # exporting
        exported_dir = self.trainer.export()
        self.logger.info(f'exported to {exported_dir}.')

        # get ready for evaluation
        exported_model = tfk.models.load_model(self.trainer.exported_saved_model_path)
        self.logger.info(f'loaded {self.trainer.exported_saved_model_path}')
        mlflow.end_run()
        eval_active_run = setup_mlflow(mlflow_tracking_uri=str(self.mlflow_tracking_uri),
                                       mlflow_experiment_name=self.project_name,
                                       base_dir=self.run_dir,
                                       evaluation=True)

        # evaluate on evaluation data
        self.logger.info('evaluating on evaluation data...')
        eval_report = self.evaluator.evaluate(data_loader=self.data_loader,
                                              preprocessor=self.preprocessor,
                                              exported_model=exported_model,
                                              active_run=eval_active_run)
        eval_report_path = self._write_eval_reports(eval_report)
        self.logger.info(f'wrote evaluation report to {eval_report_path}')
        with eval_active_run:
            mlflow.log_artifact(eval_report_path)

        # evaluate on validation data
        self.logger.info('evaluating on validation data...')
        val_index = self.data_loader.get_validation_index()
        eval_report_validation = self.evaluator.validation_evaluate(data_loader=self.data_loader,
                                                                    preprocessor=self.preprocessor,
                                                                    exported_model=exported_model,
                                                                    active_run=eval_active_run,
                                                                    index=val_index)

        val_report_path = self.run_dir.joinpath("validation_report.csv")
        eval_report_validation.to_csv(val_report_path)
        self.logger.info(f'wrote evaluation (validation dataset) to {val_report_path}')

    def _create_data_loader(self):
        try:
            class_path = self.config.data_loader_class
        except AttributeError:
            raise ConfigParamDoesNotExist('could not find data_loader_class in config file.')

        data_loader = locate(class_path)(self.config, self.data_dir)
        assert isinstance(data_loader, DataLoaderBase)
        self.logger.info('data-loader has been initialized.')
        return data_loader

    def _create_augmentor(self):
        if self.do_train_augmentation or self.do_validation_augmentation:
            try:
                class_path = self.config.augmentor_class
            except AttributeError:
                raise ConfigParamDoesNotExist('could not find augmentor_class in config file.')
        else:
            class_path = None

        if class_path is not None:
            augmentor = locate(class_path)(self.config)
            assert isinstance(augmentor, AugmentorBase)
            self.logger.info('augmentor has been initialized.')
        else:
            augmentor = None
            self.logger.info('no augmentations.')

        return augmentor

    def _create_preprocessor(self):
        try:
            class_path = self.config.preprocessor_class
        except AttributeError:
            raise ConfigParamDoesNotExist('could not find preprocessor_class in config file.')

        preprocessor = locate(class_path)(self.config)
        assert isinstance(preprocessor, PreprocessorBase)
        self.logger.info('preprocessor has been initialized.')
        return preprocessor

    def _create_model_builder(self):
        try:
            class_path = self.config.model_builder_class
        except AttributeError:
            raise ConfigParamDoesNotExist('could not find model_builder_class in config file.')

        model_builder = locate(class_path)(self.config)
        assert isinstance(model_builder, ModelBuilderBase)
        self.logger.info('model-builder has been initialized.')
        return model_builder

    def _create_trainer(self):
        trainer = Trainer(config=self.config, run_dir=self.run_dir)
        self.logger.info('trainer has been initialized')
        return trainer

    def _create_evaluator(self):
        try:
            class_path = self.config.evaluator_class
        except AttributeError:
            raise ConfigParamDoesNotExist('could not find evaluator_class in config file.')

        evaluator = locate(class_path)(self.config)
        assert isinstance(evaluator, EvaluatorBase)
        self.logger.info('evaluator has been initialized.')
        return evaluator

    def _write_eval_reports(self, report_df: pd.DataFrame) -> pathlib.Path:
        # path = EVAL_REPORTS_DIR.joinpath(self.config.project_name).joinpath(self.run_name)
        # path.mkdir(parents=True, exist_ok=True)
        eval_report_path = self.eval_report_dir.joinpath('eval_report.csv')
        report_df.to_csv(eval_report_path)
        report_df.describe().to_csv(self.eval_report_dir.joinpath('eval_report_summary.csv'))
        return eval_report_path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_dir',
                        type=str,
                        help='directory of the config file',
                        required=True)

    parser.add_argument('--data_dir',
                        type=str,
                        help='directory of the dataset',
                        required=True)

    return parser.parse_args()


class ConfigParamDoesNotExist(Exception):
    pass


if __name__ == '__main__':
    args = parse_args()

    data_dir = pathlib.Path(args.data_dir)
    run_dir = pathlib.Path(args.run_dir)

    orchestrator = Orchestrator(run_dir=run_dir, data_dir=data_dir)
    orchestrator.run()
