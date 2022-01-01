from pathlib import Path
import typing
import sys
from pydoc import locate
from dataclasses import dataclass

import mlflow
import git
import pandas as pd
import tensorflow as tf

from . import DataLoaderBase, AugmentorBase, PreprocessorBase, ModelBuilderBase, GenericModelBuilderBase, Trainer, \
    GenericTrainer, EvaluatorBase
from .utils import load_config_file, check_for_config_file, get_logger, setup_mlflow, add_config_file_to_mlflow,\
    load_config_as_dict
from model_card import ModelCardGenerator
from .abs_exceptions import *


# MLFLOW_TRACKING_URI = Path('/home').joinpath('vafaeisa').joinpath('mlruns')
# scratch_drive = Path('/home').joinpath('vafaeisa').joinpath('scratch')
# EVAL_REPORTS_DIR = scratch_drive.joinpath('eval_reports')
# PROJECT_ROOT = Path(__file__).absolute().parent.parent


@dataclass
class DataFrameReport:
    summary_df: typing.Optional[pd.DataFrame]
    path: Path


@dataclass
class Artifacts:
    exported_path: Path
    eval_report: DataFrameReport
    val_report: DataFrameReport


class Orchestrator:
    """Orchestrates pipeline components in order to train and evaluate the model.

    Notes:
        - be careful about paths, pass absolute path for ``project_root``

    Args:
        run_name (str): name of the run folder containing the config file, i.e. ``{project_root}/runs/{run_name}``
        data_dir (Optional[Path]): absolute path to dataset
        project_root (Path): absolute path to the project(repository)'s directory
        eval_reports_dir (Path): directory of evaluation reports for all projects
        mlflow_tracking_uri (Path): tracking-uri used as mlflow's backend-store

    Attributes: project_root (Path): repository's absolute path run_name (str): run's folder name run_dir (Path):
    i.e. ``{self.project_root}/runs/{self.run_name}`` data_dir (Path): absolute path to dataset logger (
    logging.Logger): orchestrator's logger object config (ConfigStruct): config_file parsed as a python object with
    nested attributes project_name (str): repository's name src_code_path (Path): absolute path to source code,
    i.e. ``{self.project_root}/{self.config.src_code_path}``, will be included in system paths eval_report_dir (
    Path): path to evaluation reports for this run mlflow_tracking_uri (Path): tracking-uri used as mlflow's
    backend-store


    """

    def __init__(self,
                 run_name: str,
                 data_dir: typing.Optional[Path],
                 project_root: Path,
                 eval_reports_dir: Path,
                 mlflow_tracking_uri: typing.Union[str, Path]):
        # Initialize the logger
        self.logger = get_logger('orchestrator')

        # Initialize paths
        self.project_root = project_root
        self.project_name = self.project_root.name
        self.src_code_path = self.project_root.joinpath(self.config.src_code_path)
        sys.path.append(str(self.src_code_path))
        self.logger.info(f'{self.src_code_path} has been added to system paths.')

        self.run_name = run_name
        self.run_dir = project_root.joinpath('runs').joinpath(run_name)
        self.logger.info(f'run directory: {self.run_dir}')

        self.exported_dir = self.run_dir.joinpath('exported')
        self.logger.info(f'will export to {self.exported_dir}')

        # Extract the code-version
        repo = git.Repo(self.project_root, search_parent_directories=False)
        self.code_version = repo.head.object.hexsha

        # Load config file
        self._load_config(data_dir)

        # Evaluation report directory
        self.eval_report_dir = eval_reports_dir.joinpath(self.project_name).joinpath(self.run_name)
        self.eval_report_dir.mkdir(parents=True, exist_ok=True)
        self.eval_report_path = self.eval_report_dir.joinpath('eval_report.csv')
        self.val_report_path = self.run_dir.joinpath("validation_report.csv")
        self.logger.info(f'evaluation reports for this run will be: {self.eval_report_path}')
        self.logger.info(f'evaluation reoprt for validation data for this run will be {self.val_report_path}')

        # Instantiating components
        self.is_tf_ = None
        self._instantiate_components()

        # Other
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.logger.info(f'MLFLow tracking uri: {self.mlflow_tracking_uri}')

        # self.mlflow_artifact_uri = mlflow.get_artifact_uri()
        # self.logger.info(f'MLFLow artifact uri: {self.mlflow_artifact_uri}')

        gpu_list = tf.config.list_physical_devices('GPU')
        self.logger.info(f'available GPU devices: {gpu_list}')

        # Artifacts
        self.artifacts = Artifacts(exported_path=self.exported_dir,
                                   eval_report=DataFrameReport(summary_df=None, path=self.eval_report_path),
                                   val_report=DataFrameReport(summary_df=None, path=self.val_report_path))

    def run(self):
        """Train, export, evaluate."""

        try:
            # Training
            self.train()
        except ExportedExists:
            self.logger.warning('model is already exported. skipping training and starting evaluation...')
        else:
            # Exporting
            self.export()

        # Evaluation
        eval_report, eval_report_val = self.evaluate()

        # Model-card generation
        self._generate_model_card(val_report=eval_report_val, eval_report=eval_report)

    def train(self):
        """train the model.

        Raises:
            ExportedExists
        """

        active_run = self._setup_mlflow_active_run(is_evaluation=False)

        # data generators
        train_data_gen, train_n = self.data_loader.create_training_generator()
        validation_data_gen, validation_n = self.data_loader.create_validation_generator()
        self.logger.info('data-generators has been created')

        # augmentation
        if self.augmentor is not None:
            if self.config.do_train_augmentation:
                train_data_gen = self.augmentor.add_augmentation(train_data_gen)
                self.logger.info('added training augmentations')
            if self.config.do_validation_augmentation:
                validation_data_gen = self.augmentor.add_augmentation(validation_data_gen)
                self.logger.info('added validation augmentations')

        # preprocessing
        train_data_gen, n_iter_train = self.preprocessor.add_preprocess(train_data_gen, train_n)
        validation_data_gen, n_iter_val = self.preprocessor.add_preprocess(validation_data_gen, validation_n)
        self.logger.info('added preprocessing to data-generators')

        # Raise exception if exported exists
        self.trainer.check_for_exported()

        self.logger.info('training started ...')
        self.trainer.train(active_run=active_run,
                           train_data_gen=train_data_gen,
                           n_iter_train=n_iter_train,
                           val_data_gen=validation_data_gen,
                           n_iter_val=n_iter_val)

    def export(self):
        self.trainer.export(dict_config=self.config_as_dict)
        self.logger.info(f'exported to {self.trainer.exported_dir}.')

    def evaluate(self) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
        """Just evaluate, if exported exists."""

        try:
            self.trainer.check_for_exported()
        except ExportedExists:
            self.trainer.load_exported()
            self.logger.info(f'evaluation will be done on this model: {self.trainer.exported_dir}')

            active_run = self._setup_mlflow_active_run(is_evaluation=True)

            eval_report_val = self._generate_eval_reports_validation(active_run)
            eval_report = self._generate_eval_reports(active_run)

            return eval_report, eval_report_val
        else:
            raise Exception(f'exported does not exist at {self.trainer.exported_dir}')

    def finalize(self):
        # TODO: commit self.artifacts to DVC
        # TODO push artifatcs to the remote
        # TODO add (logs, checkpoints) to gitignore
        # TODO commit the (model-card, .dvc files for artifacts, and .dvc/config) to git
        pass

    def _instantiate_components(self):
        self.data_loader = self._create_data_loader()
        self.augmentor = self._create_augmentor()
        self.preprocessor = self._create_preprocessor()
        self.model_builder = self._create_model_builder()
        self.trainer = self._create_trainer()
        self.evaluator = self._create_evaluator()

    def _load_config(self, data_dir: typing.Optional[Path]):
        self.config_path = check_for_config_file(self.run_dir)
        self.config = load_config_file(self.config_path.absolute())
        self.config_as_dict = load_config_as_dict(self.config_path)
        self.logger.info(f'config file loaded: {self.config_path}')

        if data_dir is not None:
            self.data_dir = data_dir
        else:
            self.data_dir = Path(self.config.data_dir)
        self.config_as_dict['data_dir'] = str(self.data_dir)
        self.config.data_dir = str(self.data_dir)
        self.logger.info(f'data directory: {self.data_dir}')

    def _generate_eval_reports_validation(self, train_active_run: mlflow.ActiveRun) -> pd.DataFrame:
        # evaluate on validation data
        self.logger.info('evaluating on validation data...')
        val_index = self.data_loader.get_validation_index()
        eval_report_validation = self.evaluator.validation_evaluate(data_loader=self.data_loader,
                                                                    preprocessor=self.preprocessor,
                                                                    model=self.model_builder,
                                                                    active_run=train_active_run,
                                                                    index=val_index)

        eval_report_validation.to_csv(self.val_report_path)
        # summary_report = eval_report_validation.describe()
        # summary_report.to_csv(self.run_dir.joinpath('val_report_summary.csv'))
        # self.artifacts.val_report.summary_df = summary_report
        self.logger.info(f'wrote evaluation (validation dataset) to {self.val_report_path}')
        # mlflow.log_artifact(str(self.val_report_path))
        return eval_report_validation

    def _generate_eval_reports(self, active_run: mlflow.ActiveRun) -> pd.DataFrame:
        """evaluate on validation and test(evaluation) data."""

        self.logger.info('evaluating on evaluation data...')
        eval_report = self.evaluator.evaluate(data_loader=self.data_loader,
                                              preprocessor=self.preprocessor,
                                              model=self.trainer.model_builder,
                                              active_run=active_run)
        eval_report.to_csv(self.eval_report_path)
        # summary_report = eval_report.describe()
        # summary_report.to_csv(self.eval_report_dir.joinpath('eval_report_summary.csv'))
        # self.artifacts.eval_report.summary_df = summary_report
        self.logger.info(f'wrote evaluation report to {self.eval_report_path}')
        # mlflow.log_artifact(str(eval_report_path))
        mlflow.end_run()

        return eval_report

    def _create_data_loader(self) -> DataLoaderBase:
        try:
            class_path = self.config.data_loader_class
        except AttributeError:
            raise ConfigParamDoesNotExist('could not find data_loader_class in config file.')

        data_loader = locate(class_path)(self.config)
        assert isinstance(data_loader, DataLoaderBase)
        self.logger.info('data-loader has been initialized.')
        return data_loader

    def _create_augmentor(self) -> typing.Optional[AugmentorBase]:
        if self.config.do_train_augmentation or self.config.do_validation_augmentation:
            try:
                class_path = self.config.augmentor_class
            except AttributeError:
                raise ConfigParamDoesNotExist('could not find augmentor_class in config file.')

            augmentor = locate(class_path)(self.config)
            assert isinstance(augmentor, AugmentorBase)
            self.logger.info('augmentor has been initialized.')
        else:
            augmentor = None
            self.logger.info('no augmentations.')

        return augmentor

    def _create_preprocessor(self) -> PreprocessorBase:
        try:
            class_path = self.config.preprocessor_class
        except AttributeError:
            raise ConfigParamDoesNotExist('could not find preprocessor_class in config file.')

        preprocessor = locate(class_path)(self.config)
        assert isinstance(preprocessor, PreprocessorBase)
        self.logger.info('preprocessor has been initialized.')
        return preprocessor

    def _create_model_builder(self) -> typing.Union[ModelBuilderBase, GenericModelBuilderBase]:
        try:
            class_path = self.config.model_builder_class
        except AttributeError:
            raise ConfigParamDoesNotExist('could not find model_builder_class in config file.')

        model_builder = locate(class_path)(self.config)
        if isinstance(model_builder, ModelBuilderBase):
            self.is_tf_ = True
        elif isinstance(model_builder, GenericModelBuilderBase):
            self.is_tf_ = False
            self.logger.info(f'model-builder is a GenericModelBuilder, switching to GenericTrainer')
        else:
            assert False, "model_builder has to be sub-classed from either ModelBuilderBase or GenericModelBuilderBase"
        self.logger.info('model-builder has been initialized.')
        return model_builder

    def _create_trainer(self) -> typing.Union[Trainer, GenericTrainer]:
        if self.is_tf_:
            trainer = Trainer(config=self.config,
                              run_dir=self.run_dir,
                              exported_dir=self.exported_dir,
                              model_builder=self.model_builder)
        else:
            trainer = GenericTrainer(config=self.config,
                                     run_dir=self.run_dir,
                                     exported_dir=self.exported_dir,
                                     model_builder=self.model_builder)
        self.logger.info('trainer has been initialized')
        return trainer

    def _create_evaluator(self) -> EvaluatorBase:
        try:
            class_path = self.config.evaluator_class
        except AttributeError:
            raise ConfigParamDoesNotExist('could not find evaluator_class in config file.')

        evaluator = locate(class_path)(self.config)
        assert isinstance(evaluator, EvaluatorBase)
        self.logger.info('evaluator has been initialized.')
        return evaluator

    def _setup_mlflow_active_run(self, is_evaluation):
        """Setup an mlflow run.

        Notes: - assumption: if ``is_evaluation``, we assume that the config file, project name, run name and code
        version are already
        """

        mlflow.end_run()
        active_run = setup_mlflow(mlflow_tracking_uri=str(self.mlflow_tracking_uri),
                                  mlflow_experiment_name=self.project_name,
                                  base_dir=self.run_dir,
                                  evaluation=is_evaluation)

        if is_evaluation:
            sess_type = 'evaluation'
        else:
            sess_type = 'training'
        mlflow.set_tag("session_type", sess_type)  # ['hpo', 'evaluation', 'training']
        try:
            add_config_file_to_mlflow(self.config_as_dict)
        except Exception as e:
            self.logger.info(f'exception when logging config file to mlflow: {e}')
        try:
            mlflow.log_param('project name', self.project_name)
        except Exception as e:
            self.logger.info(f'exception when logging project name to mlflow: {e}')
        try:
            mlflow.log_param('run name', self.run_name)
        except Exception as e:
            self.logger.info(f'exception when logging run name to mlflow: {e}')
        try:
            mlflow.log_param('code version', self.code_version)
        except Exception as e:
            self.logger.info(f'exception when logging code version to mlflow: {e}')

        self.logger.info(f'mlflow artifact-store-url for {sess_type} active-run: {mlflow.get_artifact_uri()}')
        return active_run

    def _generate_model_card(self, val_report: pd.DataFrame, eval_report: pd.DataFrame):
        """Generates model-cards.

        Notes:
            - html model-card -> ``{project_name}/runs/{run-name}/model-card/model_cards/model_card.html``
            - markdown model-card -> ``{project_name}/runs/{run-name}/README.md``
        """

        model_card_dir = self.run_dir.joinpath('model-card')
        model_card = ModelCardGenerator(model_card_dir=model_card_dir)
        self.logger.info(f'generating model-card which will be available as {model_card_dir}')
        model_card.generate(config=self.config,
                            val_eval_report=val_report,
                            eval_eval_report=eval_report)
