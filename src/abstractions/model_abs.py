from abc import ABC, abstractmethod
import typing

import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk


class BaseEstimator(ABC):
    """Dummy interface for being sure that a model conforms to ``sklearn``'s estimators syntax."""

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def score(self, x: np.ndarray, y: np.ndarray, sample_weight: typing.Optional[np.ndarray]) -> float:
        pass

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: typing.Optional[np.ndarray]) -> object:
        pass

    @classmethod
    def __subclasshook__(cls, c):
        """This defines the __subclasshook__ class method.
        This special method is called by the Python interpreter to answer the question,
         Is the class C a subclass of this class?
        """
        if cls is BaseEstimator:
            attrs = set(dir(c))
            if set(cls.__abstractmethods__) <= attrs:
                return True
        return NotImplemented


class ModelEvaluatorInterface(ABC):

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

    @classmethod
    def __subclasshook__(cls, c):
        """This defines the __subclasshook__ class method.
        This special method is called by the Python interpreter to answer the question,
         Is the class C a subclass of this class?
        """
        if cls is ModelEvaluatorInterface:
            attrs = set(dir(c))
            if set(cls.__abstractmethods__) <= attrs:
                return True
        return NotImplemented


class TensorflowModel(ModelEvaluatorInterface):

    def __init__(self, exported_model: tfk.Model):
        self.model_ = exported_model

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
        preds, gts, third_element = self.model_.predict(data_gen, steps=n_iter)
        return preds, gts, third_element

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

        return self.model_.evaluate(data_gen, steps=n_iter, return_dict=True)

    def _wrap_pred_step(self):
        """Overrides the ``predict`` method of the ``tfk.Model`` model.

        By calling ``predict`` method of the model, three lists will be returned:
         ``(predictions, ground truths, data_ids/sample_weights)``
        """

        def new_predict_step(data):
            x, y, z = tfk.utils.unpack_x_y_sample_weight(data)
            return self.model_(x, training=False), y, z

        setattr(self.model_, 'predict_step', new_predict_step)


class GenericModel(ModelEvaluatorInterface):

    def __init__(self, exported_model: BaseEstimator):
        self.model_ = exported_model

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

        x, y = list(), list()
        for i, (x_b, y_b) in enumerate(iter(data_gen)):

            x.extend(np.array(x_b).tolist())
            y.extend(np.array(y_b).tolist())

        r2_score = self.model_.score(np.array(x), np.array(y), None)
        return {'R2 Score': r2_score}

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

        y_pred, y_true, ids = list(), list(), list()

        for i, (x_b, y_b, id_b) in enumerate(iter(data_gen)):

            y_pred.extend(np.array(self.model_.predict(x_b)).tolist())
            y_true.extend(np.array(y_b).tolist())
            ids.extend(np.array(id_b).tolist())

            if i + 1 == n_iter:
                break

        return y_pred, y_true, ids
