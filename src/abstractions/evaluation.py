import types
from abc import abstractmethod

import mlflow
from tqdm import tqdm
import typing

import numpy as np
import pandas as pd
import tensorflow.keras as tfk
import tensorflow as tf

from .base_class import BaseClass
from .data_loading import DataLoaderBase
from .preprocessing import PreprocessorBase


EvalFuncs = typing.Dict[str, types.FunctionType]


class EvaluatorBase(BaseClass):

    def _load_params(self, config):
        pass

    def _set_defaults(self):
        pass

    def evaluate(self,
                 data_loader: DataLoaderBase,
                 preprocessor: PreprocessorBase,
                 exported_model: tfk.Model,
                 active_run: mlflow.ActiveRun) -> pd.DataFrame:
        """Evaluates the model using ``eval_functions`` defined in ``get_eval_functions`` on test dataset.

        Notes:
            - ``create_test_generator`` method of the ``data_loader`` will be used to create a data generator, and this method expects a tuple of ``(image, label, data_id)`` as output.

        Args:
            data_loader: will be used for creating test-data-generator
            preprocessor: will be used to add image-preprocess to the test-data-generator
            exported_model: a ``tensorflow.keras.Model`` which is ready for making inference.
            active_run: mlflow's ``ActiveRun`` instance to log evaluation reports on.

        Returns:
            a ``pandas.DataFrame`` with index= ``data_id`` s, each row represents the result for a single data-point and
            each column represents a metric.

        """

        test_data_gen, test_n = data_loader.create_test_generator()
        test_data_gen = preprocessor.add_image_preprocess(test_data_gen)
        test_data_gen = preprocessor.add_label_preprocess(test_data_gen)

        report_df = self._get_eval_report(test_data_gen, test_n, exported_model)
        self._log_to_mlflow(active_run, report_df)

        return report_df

    def validation_evaluate(self,
                            data_loader: DataLoaderBase,
                            preprocessor: PreprocessorBase,
                            exported_model: tfk.Model,
                            active_run: mlflow.ActiveRun,
                            index) -> pd.DataFrame:
        """Evaluates the model using ``eval_functions`` defined in ``get_eval_functions`` on validation dataset.

        Notes:
            - ``create_validation_generator`` method of the ``data_loader`` will be used to create a data generator, and this method expects a tuple of ``(image, label, data_id/sample_weight)`` as output.

        Args:
            data_loader: will be used for creating test-data-generator.
            preprocessor: will be used to add image-preprocess to the test-data-generator.
            exported_model: a ``tensorflow.keras.Model`` which is ready for making inference.
            active_run: mlflow's ``ActiveRun`` instance to log evaluation reports to.
            index: a list of ``data_id`` s to use as report data-frame's index. use this if your validation generator does not return a unique ``data_id`` as third element.

        Returns:
            a ``pandas.DataFrame`` with index= ``data_id`` s, each row represents the result for a single data-point
            and each column represents a metric.

        """

        val_data_gen, val_n = data_loader.create_validation_generator()
        val_data_gen = preprocessor.add_image_preprocess(val_data_gen)
        val_data_gen = preprocessor.add_label_preprocess(val_data_gen)

        report_df = self._get_eval_report(val_data_gen, val_n, exported_model)
        self._log_to_mlflow(active_run, report_df, prefix='validation')
        if index is not None:
            report_df.index = index
        else:
            report_df.index = list(range(val_n))

        return report_df

    @staticmethod
    def _wrap_pred_step(model):
        """Overrides the ``predict`` method of the model.

        By calling ``predict`` method of the model, three lists will be returned:
         ``(predictions, ground truths, data_ids/sample_weights)``
        """

        def new_predict_step(data):
            x, y, z = tfk.utils.unpack_x_y_sample_weight(data)
            return model(x, training=False), y, z

        setattr(model, 'predict_step', new_predict_step)

    @staticmethod
    def _log_to_mlflow(active_run: mlflow.ActiveRun, report_df: pd.DataFrame, prefix='test'):
        summary_report = report_df.describe()

        test_metrics = {}
        for c in summary_report.columns:
            metric_name = f'{prefix}_{c}'
            metric_value = summary_report[c]['mean']
            test_metrics[metric_name] = metric_value

        # with active_run:
        mlflow.set_tag("session_type", "evaluation")
        mlflow.log_metrics(test_metrics)

    def _get_eval_report(self, data_gen, data_n, model):
        eval_funcs = self.get_eval_funcs()
        report = {k: list() for k in eval_funcs.keys()}
        indxs = list()
        count = 0
        with tqdm(total=data_n) as pbar:
            try:
                for elem in data_gen:
                    data_id = elem[2]
                    y_true = elem[1]
                    x = elem[0]
                    y_pred = model.predict(np.expand_dims(x, axis=0))[0]

                    if isinstance(data_id, tf.Tensor):
                        indxs.append(str(data_id.numpy()))
                    else:
                        indxs.append(str(data_id))

                    for k, v in report.items():
                        v.append(eval_funcs[k](y_true, y_pred))

                    pbar.update(1)
                    count += 1
                    if count >= data_n:
                        break
            except (StopIteration, RuntimeError):
                print('stop iteration ...')

        df = pd.DataFrame(report, index=indxs)
        return df

    def _batch_eval_report(self, data_gen, n_iter, model):
        #TODO: write this method for performance improvements
        pass

    @abstractmethod
    def get_eval_funcs(self) -> EvalFuncs:
        """Evaluation functions to use for evaluation.

        You should take ``model.predict`` outputs into account. Your functions must take ``(y_true, y_pred)`` as
        inputs (not batch), and return a single output for this data-point.

        Returns:
            a dictionary mapping from metric names to metric functions: ``{'iou': iou_metric, 'dice': get_dice_metric(), ...}``

        """
