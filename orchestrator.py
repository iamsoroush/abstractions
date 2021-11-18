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

        self.data_loader, self.augmentor, self.preprocessor, self.model_builder,\
        self.trainer, self.evaluator = self._instantiate_components()

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
        mlflow_active_run = setup_mlflow(mlflow_tracking_uri=self.mlflow_tracking_uri,
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
        eval_active_run = setup_mlflow(mlflow_tracking_uri=self.mlflow_tracking_uri,
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

    def _instantiate_components(self):
        model_class_path, preprocessor_class_path, data_loader_class_path, \
        augmentor_class_path, evaluator_class_path = self._get_class_paths()

        data_loader = self._create_data_loader(data_loader_class_path)
        augmentor = self._create_augmentor(augmentor_class_path)
        preprocessor = self._create_preprocessor(preprocessor_class_path)
        model_builder = self._create_model_builder(model_class_path)
        trainer = self._create_trainer()
        evaluator = self._create_evaluator(evaluator_class_path)

        return data_loader, augmentor, preprocessor, model_builder, trainer, evaluator

    def _get_class_paths(self):
        try:
            model_class_path = self.config.model_builder_class
        except AttributeError:
            raise ConfigParamDoesNotExist('could not find model_builder_class in config file.')

        try:
            preprocessor_class_path = self.config.preprocessor_class
        except AttributeError:
            raise ConfigParamDoesNotExist('could not find preprocessor_class in config file.')

        try:
            data_loader_class_path = self.config.data_loader_class
        except AttributeError:
            raise ConfigParamDoesNotExist('could not find data_loader_class in config file.')

        if self.do_train_augmentation or self.do_validation_augmentation:
            try:
                augmentor_class_path = self.config.augmentor_class
            except AttributeError:
                raise ConfigParamDoesNotExist('could not find augmentor_class in config file.')
        else:
            augmentor_class_path = None

        try:
            evaluator_class_path = self.config.evaluator_class
        except AttributeError:
            raise ConfigParamDoesNotExist('could not find evaluator_class in config file.')

        return model_class_path, preprocessor_class_path, data_loader_class_path, augmentor_class_path, evaluator_class_path

    def _create_data_loader(self, class_path):
        data_loader = locate(class_path)(self.config, self.data_dir)
        assert isinstance(data_loader, DataLoaderBase)
        self.logger.info('data-loader has been initialized.')
        return data_loader

    def _create_augmentor(self, class_path):
        if class_path is not None:
            augmentor = locate(class_path)(self.config)
            assert isinstance(augmentor, AugmentorBase)
            self.logger.info('augmentor has been initialized.')
        else:
            augmentor = None
            self.logger.info('no augmentations.')

        return augmentor

    def _create_preprocessor(self, class_path):
        preprocessor = locate(class_path)(self.config)
        assert isinstance(preprocessor, PreprocessorBase)
        self.logger.info('preprocessor has been initialized.')
        return preprocessor

    def _create_model_builder(self, class_path):
        model_builder = locate(class_path)(self.config)
        assert isinstance(model_builder, ModelBuilderBase)
        self.logger.info('model-builder has been initialized.')
        return model_builder

    def _create_trainer(self):
        trainer = Trainer(config=self.config, run_dir=self.run_dir)
        self.logger.info('trainer has been initialized')
        return trainer

    def _create_evaluator(self, class_path):
        evaluator = locate(class_path)(self.config)
        assert isinstance(evaluator, EvaluatorBase)
        self.logger.info('evaluator has been initialized.')

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

    # config_path = check_for_config_file(run_dir)
    # config_file = load_config_file(config_path.absolute())
    # logger.info(f'config file loaded: {config_path}')
    #
    # sys.path.append(config_file.src_code_path)
    # logger.info(f'{config_file.src_code_path} has been added to system paths.')
    #
    # model_class_path, preprocessor_class_path, data_loader_class_path, \
    # augmentor_class_path, evaluator_class_path = get_class_paths(config_file)
    #
    # # Dataset
    # data_loader_class = locate(data_loader_class_path)
    # data_loader = data_loader_class(config_file, data_dir)
    # assert isinstance(data_loader, DataLoaderBase)
    # train_data_gen, train_n = data_loader.create_training_generator()
    # validation_data_gen, validation_n = data_loader.create_validation_generator()
    # logger.info('data-generators has been created')
    #
    # # Augmentor
    # if augmentor_class_path is not None:
    #     augmentor_class = locate(augmentor_class_path)
    #     augmentor = augmentor_class(config_file)
    #     assert isinstance(augmentor, AugmentorBase)
    #     if config_file.do_train_augmentation:
    #         train_data_gen = augmentor.add_augmentation(train_data_gen)
    #         logger.info('added training augmentations')
    #     if config_file.do_validation_augmentation:
    #         validation_data_gen = augmentor.add_augmentation(validation_data_gen)
    #         logger.info('added validation augmentations')
    # else:
    #     logger.info('no augmentations')
    #
    # # Preprocessor
    # preprocessor_class = locate(preprocessor_class_path)
    # preprocessor = preprocessor_class(config_file)
    # assert isinstance(preprocessor, PreprocessorBase)
    # train_data_gen, n_iter_train = preprocessor.add_preprocess(train_data_gen, train_n, config_file.batch_size)
    # validation_data_gen, n_iter_val = preprocessor.add_preprocess(validation_data_gen, validation_n, config_file.batch_size)
    # logger.info('added preprocessing to data-generators')
    #
    # # Model
    # model_class = locate(model_class_path)
    # model_builder = model_class(config_file)
    # logger.info('model-builder has been created')
    #
    # # Trainer
    # mlflow_active_run = setup_mlflow(mlflow_tracking_uri=MLFLOW_TRACKING_URI,
    #                                  mlflow_experiment_name=config_file.project_name,
    #                                  base_dir=run_dir)
    # trainer = Trainer(config=config_file, run_dir=run_dir)
    # logger.info('trainer has been initialized')
    #
    # # Train
    # logger.info('training started ...')
    # trainer.train(model_builder=model_builder,
    #               active_run=mlflow_active_run,
    #               train_data_gen=train_data_gen,
    #               n_iter_train=n_iter_train,
    #               val_data_gen=validation_data_gen,
    #               n_iter_val=n_iter_val)
    #
    # # Export
    # exported_dir = trainer.export()
    # logger.info(f'exported to {exported_dir}.')
    #
    # # Initializations for evaluation
    # exported_model = tfk.models.load_model(trainer.exported_saved_model_path)
    # logger.info(f'loaded {trainer.exported_saved_model_path}')
    # mlflow.end_run()
    # eval_active_run = setup_mlflow(mlflow_tracking_uri=MLFLOW_TRACKING_URI,
    #                                mlflow_experiment_name=config_file.project_name,
    #                                base_dir=run_dir,
    #                                evaluation=True)
    # evaluator_class = locate(evaluator_class_path)
    # evaluator = evaluator_class(config_file)
    # assert isinstance(evaluator, EvaluatorBase)
    #
    # # Evaluate on evaluation data
    # logger.info('evaluation started...')
    # eval_report = evaluator.evaluate(data_loader=data_loader,
    #                                  preprocessor=preprocessor,
    #                                  exported_model=exported_model,
    #                                  active_run=eval_active_run)
    # eval_report_path = write_eval_reports(eval_report, str(config_file.project_name), run_dir.name)
    # logger.info(f'wrote evaluation report to {eval_report_path}')
    # with eval_active_run:
    #     mlflow.log_artifact(eval_report_path)
    #
    # # Evaluate on validation data
    # logger.info('evaluation (validation data) started...')
    # val_index = data_loader.get_validation_index()
    # eval_report_validation = evaluator.validation_evaluate(data_loader=data_loader,
    #                                                        preprocessor=preprocessor,
    #                                                        exported_model=exported_model,
    #                                                        active_run=eval_active_run,
    #                                                        index=val_index)
    # eval_report_validation.to_csv(run_dir.joinpath("validation_report.csv"))
    # logger.info(f'wrote evaluation (validation dataset) to {run_dir.joinpath("validation_report.csv")}')
