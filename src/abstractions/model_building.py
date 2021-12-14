from abc import abstractmethod
import typing
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
from sklearn.base import BaseEstimator

from .base_class import BaseClass


class MBBase(BaseClass):

    def __init__(self, config):
        super().__init__(config=config)
        self.model_ = None

    @abstractmethod
    def evaluate(self,
                 data_gen: typing.Iterator,
                 n_iter: int) -> dict:
        """Evaluation.

        Args:
            data_gen (Iterator): batch data generator, which yields (x_batch, y_batch) in each iteration.
            n_iter (int): iterations needed to entirely cover the data-samples

        Returns:
            a dictionary containing averaged values across entire ``data_gen`` for internal metrics of the model, i.e. {met1: averaged_val, met2: averaged_val, ...}
        """

    @abstractmethod
    def predict(self,
                data_gen: typing.Iterator,
                n_iter: int) -> typing.Tuple[typing.Union[np.ndarray, tf.Tensor],
                                            typing.Union[np.ndarray, tf.Tensor],
                                            typing.Union[np.ndarray, tf.Tensor, list]]:
        """Predict on a data generator which generates batches of data in each iteration.

        Args:
            data_gen (Iterator): batch data generator, which yields (x_batch, y_batch, sample_id_batch) in each iteration.
            n_iter (int): iterations needed to entirely cover the data-samples

        Returns:
            tuple(predictions, labels, data_ids):
            - predictions: predictions of the model for each data-sample, of ``shape(n_samples, ...)
            - labels: labels for each data-sample, has the same shape as ``predictions``
            - data_ids: data-id for each sample, of ``shape(n_samples,)``
        """

    @abstractmethod
    def load(self, exported_path: Path):
        pass


class ModelBuilderBase(MBBase):
    """Building and compiling ``tensorflow.keras`` models to train with Trainer.

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

    def evaluate(self,
                 data_gen: typing.Iterator,
                 n_iter: int) -> dict:
        """Evaluation.

        Args:
            data_gen (Iterator): batch data generator, which yields (x_batch, y_batch) in each iteration.
            n_iter (int): iterations needed to entirely cover the data-samples

        Returns:
            a dictionary containing averaged values across entire ``data_gen`` for internal metrics of the model, i.e. {met1: averaged_val, met2: averaged_val, ...}
        """

        return self.model.evaluate(data_gen, steps=n_iter, return_dict=True)

    def predict(self,
                data_gen: typing.Iterator,
                n_iter: int) -> typing.Tuple[typing.Any, typing.Any, typing.Any]:
        """Predict on a data generator which generates batches of data in each iteration.

        Args:
            data_gen (Iterator): batch data generator, which yields (x_batch, y_batch, sample_id_batch) in each iteration.
            n_iter (int): iterations needed to entirely cover the data-samples

        Returns:
            tuple(predictions, labels, data_ids):
            - predictions: predictions of the model for each data-sample, of ``shape(n_samples, ...)
            - labels: labels for each data-sample, has the same shape as ``predictions``
            - data_ids: data-id for each sample, of ``shape(n_samples,)``
        """

        self._wrap_pred_step()
        preds, gts, third_element = self.model.predict(data_gen, steps=n_iter)
        return preds, gts, third_element

    def load(self, exported_path: Path) -> tfk.Model:
        return tfk.models.load_model(exported_path)

    def _wrap_pred_step(self):
        """Overrides the ``predict`` method of the ``tfk.Model`` model.

        By calling ``predict`` method of the model, three lists will be returned:
         ``(predictions, ground truths, data_ids/sample_weights)``
        """

        def new_predict_step(data):
            x, y, z = tfk.utils.unpack_x_y_sample_weight(data)
            return self.model(x, training=False), y, z

        setattr(self.model, 'predict_step', new_predict_step)
    #
    # @abstractmethod
    # def post_process(self, y_pred):
    #     """Define your post-processing, used for evaluation.
    #
    #     If you have ``softmax`` as your final layer, but your labels are sparse-categorical labels, you will need to
    #      post-process the output of your model before comparing it to ``y_true``. In this case you should use
    #      ``return np.argmax(y_pred)``.
    #
    #      Note: make sure that this is compatible with your evaluation functions and ground truth labels generated
    #       using ``DataLoader``'s generators.
    #
    #      Args:
    #          y_pred: a tensor generated by ``model.predict`` method. Note that the first dimension is ``batch``.
    #
    #     Returns:
    #         post-processed batch of y_pred, ready to be compared to ground-truth.
    #
    #     """

    @property
    def model(self) -> tfk.Model:
        return self.model_

    @model.setter
    def model(self, new_model):
        self.model_ = new_model

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

    @abstractmethod
    def fit(self,
            train_data_gen: typing.Iterator,
            n_iter_train: int,
            val_data_gen: typing.Iterator,
            n_iter_val: int) -> BaseEstimator:
        pass

    @abstractmethod
    def evaluate(self,
                 data_gen: typing.Iterator,
                 n_iter: int) -> dict:
        """Evaluation.

        Args:
            data_gen (Iterator): batch data generator, which yields (x_batch, y_batch) in each iteration.
            n_iter (int): iterations needed to entirely cover the data-samples

        Returns:
            a dictionary containing averaged values across entire ``data_gen`` for internal metrics of the model, i.e. ``.score`` method's result for ``sklearn`` models.
        """

    @staticmethod
    def export(exported_dir: Path,
               model: BaseEstimator):
        """Exports (writes) the model to the given path, and returns the exported model.

        Notes:
            - you can use ``joblib.dump(model, exported_dir.joinpath('exported.joblib'))``

        """

        export_file = exported_dir.joinpath('model.joblib')
        joblib.dump(model, str(export_file))

    def load(self, exported_dir: Path) -> BaseEstimator:
        export_file = exported_dir.joinpath('model.joblib')
        loaded = joblib.load(str(export_file))
        return loaded

    @abstractmethod
    def predict(self,
                data_gen: typing.Iterator,
                n_iter: int) -> typing.Tuple[typing.Any, typing.Any, typing.Any]:
        """Predict on a data generator which generates batches of data in each iteration.

        Args:
            data_gen (Iterator): batch data generator, which yields (x_batch, y_batch, sample_id_batch) in each iteration.
            n_iter (int): iterations needed to entirely cover the data-samples

        Returns:
            tuple(predictions, labels, data_ids):
            - predictions: predictions of the model for each data-sample, of ``shape(n_samples, ...)
            - labels: labels for each data-sample, has the same shape as ``predictions``
            - data_ids: data-id for each sample, of ``shape(n_samples,)``
        """

    @property
    def model(self) -> BaseEstimator:
        return self.model_

    @model.setter
    def model(self, new_model):
        self.model_ = new_model
