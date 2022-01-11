from abc import abstractmethod, ABC
import typing
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk

from .base_class import BaseClass
from .model_abs import BaseEstimator, TensorflowModel, GenericModel, ModelEvaluatorInterface


class MBBase(BaseClass, ABC):

    def load(self, exported_path: Path) -> ModelEvaluatorInterface:
        pass


class ModelBuilderBase(MBBase):
    """Building and compiling ``tensorflow.keras`` models to train with Trainer.

    Notes:
        - you have to override these methods: ``get_compiled_model``
        - you may override these methods too (optional): ``get_callbacks``, ``get_class_weight``
        - don't override the private ``_{method_name}`` methods
        - don't override these methods: ``

    Examples:
        >>> model_builder = ModelBuilder(config)
        >>> model = model_builder.get_compiled_model()
        >>> callbacks = model_builder.get_callbacks()
        >>> class_weight = model_builder.get_class_weight()
        >>> model.fit(train_gen, n_iter_train, callbacks=callbacks, class_weight=class_weight)

    """

    @abstractmethod
    def get_compiled_model(self) -> tfk.Model:
        """Generates the model for training, and returns the compiled model.

        Returns:
            A compiled ``tensorflow.keras`` model.
        """

        pass

    def get_callbacks(self) -> list:
        """Returns any callbacks for ``fit``.

        Returns:
            list of ``tf.keras.Callback`` objects. ``Orchestrator`` will handle the ``ModelCheckpoint`` and ``Tensorboard`` callbacks.
            Still, you can return each of these two callbacks, and orchestrator will modify your callbacks if needed.

        """
        return list()

    def get_class_weight(self) -> typing.Optional[dict]:
        """Set this if you want to pass ``class_weight`` to ``fit``.

        Returns:
           Optional dictionary mapping class indices (integers) to a weight (float) value.
           used for weighting the loss function (during training only).
           This can be useful to tell the model to "pay more attention" to samples from an under-represented class.

        """

        return None

    def load(self, exported_path: Path) -> TensorflowModel:
        exported_model = tfk.models.load_model(exported_path)
        return TensorflowModel(exported_model=exported_model)

    @classmethod
    def __subclasshook__(cls, c):
        """This defines the __subclasshook__ class method.
        This special method is called by the Python interpreter to answer the question,
         Is the class C a subclass of this class?
        """

        if cls is ModelBuilderBase:
            attrs = set(dir(c))
            if set(cls.__abstractmethods__) <= attrs:
                return True
        return NotImplemented


class GenericModelBuilderBase(MBBase):
    """Model-builder for ``sklearn`` models to be trained using ``GenericTrainer``.

    Notes:
        - you have to override these methods: ``.fit``

    Examples:
        >>> model_builder = GenericModelBuilder(config)
        >>> model_builder.fit(train_gen, n_iter_train, val_gen, n_iter_val)
        >>> callbacks = model_builder.get_callbacks()
        >>> class_weight = model_builder.get_class_weight()
        >>> model.fit(train_gen, n_iter_train, callbacks=callbacks, class_weight=class_weight)

    """

    def __init__(self, config):
        super().__init__(config=config)
        self.model_file_name = 'model.joblib'

    @abstractmethod
    def fit(self,
            train_data_gen: typing.Iterator,
            n_iter_train: int,
            val_data_gen: typing.Iterator,
            n_iter_val: int) -> BaseEstimator:
        """Fit/train your model here, and return the fitted model

        """

    def load(self, exported_dir: Path) -> GenericModel:
        export_file = exported_dir.joinpath(self.model_file_name)
        loaded = self._load_from_path(export_file)
        return GenericModel(loaded)

    @staticmethod
    def _load_from_path(exported_model_path: Path) -> BaseEstimator:
        loaded = joblib.load(str(exported_model_path))
        return loaded
