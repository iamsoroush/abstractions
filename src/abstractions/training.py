import os
import typing
from abc import ABC, abstractmethod
import yaml

import shutil
from pathlib import Path

import tensorflow.keras as tfk
import tensorflow.keras.callbacks as tfkc
import mlflow

from .base_class import BaseClass
from .model_building import ModelBuilderBase, GenericModelBuilderBase
from .abs_exceptions import *
from .utils import ConfigStruct


class TrainerBase(BaseClass, ABC):
    """

    Args:
        model_builder: an instance of ``MBBase``
    """

    def __init__(self,
                 config: ConfigStruct,
                 run_dir: Path,
                 exported_dir: Path,
                 model_builder: typing.Union[ModelBuilderBase, GenericModelBuilderBase]):
        self.run_dir = Path(run_dir)
        self.exported_dir = exported_dir
        super().__init__(config=config)

        # Paths
        self.run_id_path = self.run_dir.joinpath('run_id.txt')
        self.config_file_path = [i for i in self.run_dir.iterdir() if i.name.endswith('.yaml')][0]
        self.model_builder = model_builder

    @abstractmethod
    def train(self,
              active_run: mlflow.ActiveRun,
              train_data_gen,
              n_iter_train,
              val_data_gen,
              n_iter_val):
        """Training the model."""

    @abstractmethod
    def export(self, dict_config: dict):
        """Exports the model to ``self.exported_dir``.

        Args:
            dict_config: config file as a nested dictionary to write in exported folder
        """

    @abstractmethod
    def load_exported(self):
        """Loads exported model into ``self.model_builder.model``."""

    @abstractmethod
    def check_for_exported(self):
        """Checks for existence of exported model, if so, raises ``ExportedExists`` exception."""

    def _write_mlflow_run_id(self, run: mlflow.ActiveRun):
        with open(self.run_id_path, 'w') as f:
            f.write(run.info.run_id)

    @staticmethod
    def _write_dict_to_yaml(data: dict, path: Path):
        with open(path, 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)


class Trainer(TrainerBase):
    """Trainer for ``tensorflow.keras`` models."""

    def _load_params(self, config):
        self.epochs = config.epochs
        self.export_metric = config.export.metric
        self.export_mode = config.export.mode

    def _set_defaults(self):
        self.epochs = 10
        self.export_metric = 'val_loss'
        self.export_mode = 'min'

    def __init__(self,
                 config: ConfigStruct,
                 run_dir: Path,
                 exported_dir: Path,
                 model_builder: ModelBuilderBase):
        super().__init__(config=config, run_dir=run_dir, exported_dir=exported_dir, model_builder=model_builder)

        # Paths
        self.checkpoints_dir = self.run_dir.joinpath('checkpoints')
        self.tensorboard_log_dir = self.run_dir.joinpath('logs')
        self.exported_saved_model_path = self.exported_dir.joinpath('savedmodel')

        # Make logs and checkpoints directories
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.tensorboard_log_dir.mkdir(exist_ok=True)

        # Container for fit_history
        self.history_ = None

    def train(self,
              active_run: mlflow.ActiveRun,
              train_data_gen,
              n_iter_train,
              val_data_gen,
              n_iter_val):
        """Trains the model using data generators and logs to ``mlflow``.

        Will try to resume training from latest checkpoint, else starts training from ``epoch=0``.

        Args:
            active_run: ``mlflow.ActiveRun`` for logging to **mlflow**.
            train_data_gen: preprocessed, augmented training-data-generator.This will be the output of
             ``Preprocessor.add_preprocess``.
            n_iter_train: ``steps_per_epoch`` for ``model.fit``
            val_data_gen: preprocessed, augmented validation data-generator.This will be the output of
             ``Preprocessor.add_preprocess``.
            n_iter_val: ``validation_steps`` for ``model.fit``

        """

        # # Raise exception if the best model is already exported.
        # self.check_for_exported()

        # Create/load model and define initial_epoch
        if any(self._get_checkpoints()):
            model, initial_epoch = self._load_latest_model()
        else:
            initial_epoch = 0
            model = self.model_builder.get_compiled_model()

        # Get callbacks
        callbacks = self._get_callbacks(self.model_builder)

        # Write run_id
        self._write_mlflow_run_id(active_run)

        # Enable autolog
        mlflow.tensorflow.autolog(every_n_iter=1,
                                  log_models=False,
                                  disable=False,
                                  exclusive=False,
                                  disable_for_unsupported_versions=True,
                                  silent=False)
        # mlflow.autolog(log_models=False)
        # mlflow.keras.autolog(log_models=False)

        # Fit
        model.fit(train_data_gen,
                  steps_per_epoch=n_iter_train,
                  initial_epoch=initial_epoch,
                  epochs=self.epochs,
                  validation_data=val_data_gen,
                  validation_steps=n_iter_val,
                  class_weight=self.model_builder.get_class_weight(),
                  callbacks=callbacks)

    def export(self, dict_config: dict):
        """Exports the best version of ``SavedModel`` s, and ``config.yaml`` file into exported sub_directory.

        This method will delete all checkpoints after exporting the best one.

        Args:
            dict_config: config file as a nested dictionary to write in exported folder
        """

        best_model_info = self._get_best_checkpoint()

        # exported_saved_model_path = self.exported_dir.joinpath('savedmodel')
        exported_config_path = self.exported_dir.joinpath('config.yaml')
        shutil.copytree(best_model_info['path'], self.exported_saved_model_path,
                        symlinks=False, ignore=None, ignore_dangling_symlinks=False)
        # shutil.copy(self.config_file_path, exported_config_path)
        self._write_dict_to_yaml(dict_config, exported_config_path)

        # Delete checkpoints
        shutil.rmtree(self.checkpoints_dir)

        # # Load the exported SavedModel
        # exported_model = tfk.models.load_model(self.exported_saved_model_path)
        # model_builder.model = exported_model

    def load_exported(self):
        """Loads exported model into ``self.model_builder.model``."""

        exported_model = self.model_builder._load(self.exported_saved_model_path)
        self.model_builder._model = exported_model

    def check_for_exported(self):
        """Raises exception if exported directory exists and contains ``savedmodel``"""

        if self.exported_dir.is_dir():
            if any(self.exported_dir.iterdir()):
                if self.exported_saved_model_path.exists():
                    raise ExportedExists('exported files already exist.')

    def _get_callbacks(self, model_builder: ModelBuilderBase):
        """Makes sure that TensorBoard and ModelCheckpoint callbacks exist and are correctly configured.

        Attributes:
            model_builder: ``ModelBuilder`` object, to get callbacks list using ``model_builder.get_callbacks``

        modifies ``callbacks`` to be a list of callbacks, in which ``TensorBoard`` callback exists with
         ``log_dir=self.tensorboard_log_dir`` and ``ModelCheckpoint`` callback exists with
          ``filepath=self.checkpoints_dir/...``, ``save_weights_only=False``

        """

        callbacks = model_builder.get_callbacks()

        mc_callbacks = [i for i in callbacks if isinstance(i, tfkc.ModelCheckpoint)]
        tb_callbacks = [i for i in callbacks if isinstance(i, tfkc.TensorBoard)]

        to_track = self.export_metric
        checkpoint_path = str(self.checkpoints_dir) + "/sm-{epoch:04d}"
        checkpoint_path = checkpoint_path + "-{" + to_track + ":4.5f}"

        if any(mc_callbacks):
            mc_callbacks[0].filepath = str(checkpoint_path)
            mc_callbacks[0].save_weights_only = False
        else:
            mc = tfkc.ModelCheckpoint(filepath=checkpoint_path,
                                      save_weights_only=False)
            callbacks.append(mc)

        if any(tb_callbacks):
            tb_callbacks[0].log_dir = self.tensorboard_log_dir
        else:
            tb = tfkc.TensorBoard(log_dir=self.tensorboard_log_dir)
            callbacks.append(tb)

        return callbacks

    def _load_latest_model(self):
        """Loads and returns the latest ``SavedModel``.

        Returns:
            model: a ``tf.keras.Model`` object.
            initial_epoch: initial epoch for this checkpoint

        """

        latest_ch = self._get_latest_checkpoint()
        initial_epoch = latest_ch['epoch']
        sm_path = latest_ch['path']
        print(f'found latest checkpoint: {sm_path}')
        print(f'resuming from epoch {initial_epoch}')
        model = tfk.models.load_model(latest_ch['path'])
        return model, initial_epoch

    def _get_latest_checkpoint(self):
        """Returns info about the latest checkpoint.

        Returns:
            a dictionary containing epoch, path to ``SavedModel`` and value of ``self.export_metric`` for
            latest checkpoint:
                {'epoch': int, 'path': pathlib.Path, 'value': float}

        """

        checkpoints = self._get_checkpoints_info()
        return max(checkpoints, key=lambda x: os.path.getctime(x['path']))

    def _get_best_checkpoint(self):
        """Returns info about the best checkpoint.

        Returns:
            a dictionary containing epoch, path to ``SavedModel`` and value of ``self.export_metric`` for
            the best checkpoint in terms of ``self.export_metric``:
                {'epoch': int, 'path': pathlib.Path, 'value': float}

        """

        checkpoints = self._get_checkpoints_info()

        if self.export_mode == 'min':
            selected_model = min(checkpoints, key=lambda x: x['value'])
        else:
            selected_model = max(checkpoints, key=lambda x: x['value'])
        return selected_model

    def _get_checkpoints(self):
        """Returns a list of paths to folders containing a ``saved_model.pb``"""

        ckpts = [item for item in self.checkpoints_dir.iterdir() if any(item.glob('saved_model.pb'))]
        return ckpts

    def _get_checkpoints_info(self):
        """Returns info about checkpoints.

        Returns:
            A list of dictionaries related to each checkpoint:
                {'epoch': int, 'path': pathlib.Path, 'value': float}

        """

        checkpoints = self._get_checkpoints()
        ckpt_info = list()
        for cp in checkpoints:
            splits = str(cp.name).split('-')
            epoch = int(splits[1])
            metric_value = float(splits[2])
            ckpt_info.append({'path': cp, 'epoch': epoch, 'value': metric_value})
        return ckpt_info


class GenericTrainer(TrainerBase):
    """Generic trainer for models that aim to define the training inside themselves, e.g. sklearn models.

    Notes:
        - this trainer will call the ``.train`` method of the model
    """

    def __init__(self,
                 config: ConfigStruct,
                 run_dir: Path,
                 exported_dir: Path,
                 model_builder: GenericModelBuilderBase):
        super().__init__(config=config, run_dir=run_dir, exported_dir=exported_dir, model_builder=model_builder)
        self.model_ = None
        self.exported_model_path = None

    def _set_defaults(self):
        pass

    def _load_params(self, config):
        pass

    def train(self,
              active_run: mlflow.ActiveRun,
              train_data_gen,
              n_iter_train,
              val_data_gen,
              n_iter_val):
        """Trains the model using data generators and logs to ``mlflow``.


        Args:
            active_run: ``mlflow.ActiveRun`` for logging to **mlflow**.
            train_data_gen: preprocessed, augmented training-data-generator.This will be the output of the
             ``Preprocessor.add_preprocess``.
            n_iter_train: number of iterations needed to get all data from ``train_data_gen``
            val_data_gen: preprocessed, augmented validation data-generator.This will be the output of the
             ``Preprocessor.add_preprocess``.
            n_iter_val: number of iterations needed to get all data from ``val_data_gen``

        """

        # Raise exception if the model is already exported, i.e. the ``self.exported_dir`` is not empty.
        self.check_for_exported()

        # Write run_id
        self._write_mlflow_run_id(active_run)

        mlflow.autolog(log_models=False)
        fitted_model = self.model_builder.fit(train_data_gen,
                                              n_iter_train,
                                              val_data_gen,
                                              n_iter_val)
        self.model_ = fitted_model

    def export(self, dict_config: dict):
        """Exports the trained model, and ``config.yaml`` file into exported sub_directory, and adds

        Args:
            dict_config: config file as a nested dictionary to write in exported folder

        """

        exported_config_path = self.exported_dir.joinpath('config.yaml')
        # shutil.copy(self.config_file_path, exported_config_path)
        self._write_dict_to_yaml(dict_config, exported_config_path)
        self.exported_model_path = self.model_builder._export(self.exported_dir, self.model_)

    def load_exported(self):
        """Loads exported model into ``self.model_builder.model``."""

        loaded = self.model_builder._load(self.exported_dir)
        self.model_builder._model = loaded

    def check_for_exported(self):
        """Raises exception if exported directory exists and is not empty."""

        if self.exported_dir.is_dir():
            if any(self.exported_dir.iterdir()):
                raise ExportedExists('exported files already exist.')
