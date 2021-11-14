import types
from abc import abstractmethod

import pandas
from tqdm import tqdm
import typing

import numpy as np
import pandas as pd
import tensorflow as tf

from .base_class import BaseClass
from .data_loader import DataLoaderBase
from .preprocessor import PreprocessorBase


EvalFuncs = typing.Dict[str, types.FunctionType]


class EvaluatorBase(BaseClass):

    def _load_params(self, config):
        pass

    def _set_defaults(self):
        pass

    def evaluate(self,
                 data_loader: DataLoaderBase,
                 preprocessor: PreprocessorBase,
                 exported_model) -> pandas.DataFrame:
        """Evaluates the model using ``eval_functions`` defined in ``get_eval_functions``.

        Note: ``create_test_generator`` method of the ``data_loader`` will be used to create a data generator,
         and this method expects an output of ``(image, label, data_id)`` as the output.

        Args:
            data_loader: DataLoaderBase, will be used for creating test-data-generator
            preprocessor: PreprocessorBase, will be used to add image-preprocess to the test-data-generator
            exported_model: a tensorflow.keras.Model which is ready for making inference.

        Returns:
            df: a pandas.DataFrame with index as ``data_id``s, each row represents the result for a single data-point
             and each column represents a metric.

        """

        test_data_gen, test_n = data_loader.create_test_generator()
        test_data_gen = preprocessor.add_image_preprocess(test_data_gen)

        eval_funcs = self.get_eval_funcs()
        report = {k: list() for k in eval_funcs.keys()}
        indxs = list()
        count = 1
        with tqdm(total=test_n) as pbar:
            try:
                for elem in test_data_gen:
                    data_id = elem[2]
                    y_true = elem[1]
                    x = elem[0]
                    y_pred = exported_model.predict(np.expand_dims(x, axis=0))[0]

                    if isinstance(data_id, tf.Tensor):
                        indxs.append(str(data_id.numpy()))
                    else:
                        indxs.append(str(data_id))

                    for k, v in report.items():
                        v.append(eval_funcs[k](y_true, y_pred))

                    pbar.update(1)
                    count += 1
                    if count > test_n:
                        break
            except (StopIteration, RuntimeError):
                print('stop iteration ...')

        df = pd.DataFrame(report, index=indxs)
        return df

    @abstractmethod
    def get_eval_funcs(self) -> EvalFuncs:
        """Evaluation functions to use for evaluation.

        You should take ``model.predict``'s outputs into account. Your functions should take ``(y_true, y_pred)`` as
         inputs (not batch), and return a single output for this data-point.

        Returns:
            a dictionary mapping from metric names to metric functions:
                {'iou': iou_metric,
                 'dice': get_dice_metric(), ...}

        """

    @classmethod
    def __subclasshook__(cls, c):
        """This defines the __subclasshook__ class method.
        This special method is called by the Python interpreter to answer the question,
         Is the class C a subclass of this class?
        """

        if cls is EvaluatorBase:
            attrs = set(dir(c))
            if set(cls.__abstractmethods__) <= attrs:
                return True
        return NotImplemented
