import types
from abc import abstractmethod

import mlflow
from tqdm import tqdm
import typing
import yaml

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
            - ``create_test_generator`` method of the ``data_loader`` will be used to create a data generator, and it is expected to have a tuple of ``(image, label, data_id)`` as output.

        Args:
            data_loader: will be used for creating test-data-generator
            preprocessor: will be used to add image-preprocess to the test-data-generator
            exported_model: a ``tensorflow.keras.Model`` which is ready for making inference.
            active_run: mlflow's ``ActiveRun`` instance to log evaluation reports on.

        Returns:
            a ``pandas.DataFrame`` with index= ``data_id`` s, each row represents the result for a single data-point and
            each column represents a metric.

        """

        def wrapper_gen(gen):
            while True:
                x_b, y_b, w_b = next(gen)
                yield x_b, y_b

        # Internal metrics + loss
        test_data_gen, test_n = data_loader.create_test_generator()
        test_data_gen = preprocessor.add_image_preprocess(test_data_gen)
        test_data_gen = preprocessor.add_label_preprocess(test_data_gen)
        test_data_gen, n_iter_test = preprocessor.batchify(test_data_gen, test_n)
        eval_internal_metrics = dict()
        for k, v in exported_model.evaluate(iter(wrapper_gen(iter(test_data_gen))),
                                            steps=n_iter_test, return_dict=True).items():
            eval_internal_metrics[f'_model.evaluate_{k}'] = v
        self._log_metrics_to_mlflow(active_run, eval_internal_metrics, prefix='test')

        # Eval funcs
        test_data_gen, test_n = data_loader.create_test_generator()
        test_data_gen = preprocessor.add_image_preprocess(test_data_gen)
        test_data_gen = preprocessor.add_label_preprocess(test_data_gen)
        test_data_gen, n_iter_test = preprocessor.batchify(test_data_gen, test_n)

        preds, gts, data_ids = self._get_model_outputs(exported_model, test_data_gen, n_iter_test)
        report_df = self._generate_eval_reports_test(preds, gts, data_ids)
        self._log_df_report_to_mlflow(active_run, report_df, prefix='test')

        # report_df = self._get_eval_report(test_data_gen, test_n, exported_model)
        # self._log_df_report_to_mlflow(active_run, report_df)

        return report_df

    def validation_evaluate(self,
                            data_loader: DataLoaderBase,
                            preprocessor: PreprocessorBase,
                            exported_model: tfk.Model,
                            active_run: typing.Optional[mlflow.ActiveRun],
                            index) -> pd.DataFrame:
        """Evaluates the model using ``eval_functions`` defined in ``get_eval_functions`` on validation dataset.

        Notes:
            - ``create_validation_generator`` method of the ``data_loader`` will be used to create a data generator which yields a tuple of ``(image, label, sample_weight)`` as output.

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

        # Internal eval metrics + loss value
        val_data_gen, val_n = data_loader.create_validation_generator()
        val_data_gen, n_iter_val = preprocessor.add_preprocess(val_data_gen, n_data_points=val_n)
        eval_internal_metrics = dict()
        for k, v in exported_model.evaluate(val_data_gen, steps=n_iter_val, return_dict=True).items():
            eval_internal_metrics[f'_model.evaluate_{k}'] = v
        self._log_metrics_to_mlflow(active_run, eval_internal_metrics, prefix='validation')

        # Eval funcs
        val_data_gen, val_n = data_loader.create_validation_generator()
        val_data_gen, n_iter_val = preprocessor.add_preprocess(val_data_gen, n_data_points=val_n)
        # indxs = data_loader.get_validation_index()
        preds, gts, _ = self._get_model_outputs(exported_model, val_data_gen, n_iter_val)
        report_df = self._generate_eval_reports_val(preds, gts, index)
        self._log_df_report_to_mlflow(active_run, report_df, prefix='validation')
        # if index is not None:
        #     report_df.index = index
        # else:
        #     report_df.index = list(range(val_n))

        return report_df

    def _get_model_outputs(self, model, data_gen, n_iter):
        self._wrap_pred_step(model)
        preds, gts, third_element = model.predict(data_gen, steps=n_iter)
        return preds, gts, third_element

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
    def _log_df_report_to_mlflow(active_run: typing.Optional[mlflow.ActiveRun], report_df: pd.DataFrame, prefix='test'):
        if active_run is not None:
            summary_report = report_df.describe()

            test_metrics = {}
            for c in summary_report.columns:
                metric_name = f'{prefix}_{c}'
                metric_value = summary_report[c]['mean']
                test_metrics[metric_name] = metric_value

            mlflow.log_metrics(test_metrics)

    @staticmethod
    def _log_metrics_to_mlflow(active_run: typing.Optional[mlflow.ActiveRun], metrics: dict, prefix='test'):
        if active_run is not None:

            metrics_to_log = {}
            for k, v in metrics.items():
                metric_name = f'{prefix}_{k}'
                metric_value = v
                metrics_to_log[metric_name] = metric_value
            mlflow.log_metrics(metrics_to_log)

    def _generate_eval_reports_val(self, preds, gts, indxs):
        eval_funcs = self.get_eval_funcs()
        report = {k: list() for k in eval_funcs.keys()}
        n_data = len(preds)
        if indxs is None:
            indxs = list(range(n_data))

        with tqdm(total=n_data) as pbar:
            for ind, (y_pred, y_true) in enumerate(zip(preds, gts)):
                for k, v in report.items():
                    metric_val = eval_funcs[k](y_true, y_pred)
                    if isinstance(metric_val, tf.Tensor):
                        v.append(metric_val.numpy())
                    else:
                        v.append(metric_val)
                pbar.update(1)

        df = pd.DataFrame(report, index=indxs)
        return df

    def _generate_eval_reports_test(self, preds, gts, data_ids):
        eval_funcs = self.get_eval_funcs()
        report = {k: list() for k in eval_funcs.keys()}
        n_data = len(preds)
        indxs = list()
        with tqdm(total=n_data) as pbar:
            for ind, (y_pred, y_true, data_id) in enumerate(zip(preds, gts, data_ids)):
                for k, v in report.items():
                    metric_val = eval_funcs[k](y_true, y_pred)
                    if isinstance(metric_val, tf.Tensor):
                        v.append(metric_val.numpy())
                    else:
                        v.append(metric_val)
                if hasattr(data_id, '__iter__'):
                    indxs.append(data_id[0])
                else:
                    if isinstance(data_id, tf.Tensor):
                        indxs.append(data_id.numpy())
                    else:
                        indxs.append(data_id)
                pbar.update(1)

        df = pd.DataFrame(report, index=indxs)
        return df

    # def _get_eval_report(self, data_gen, data_n, model):
    #     eval_funcs = self.get_eval_funcs()
    #     report = {k: list() for k in eval_funcs.keys()}
    #     indxs = list()
    #     count = 0
    #     with tqdm(total=data_n) as pbar:
    #         try:
    #             for elem in data_gen:
    #                 data_id = elem[2]
    #                 y_true = elem[1]
    #                 x = elem[0]
    #                 y_pred = model.predict(np.expand_dims(x, axis=0))[0]
    #
    #                 if isinstance(data_id, tf.Tensor):
    #                     indxs.append(str(data_id.numpy()))
    #                 else:
    #                     indxs.append(str(data_id))
    #
    #                 for k, v in report.items():
    #                     v.append(eval_funcs[k](y_true, y_pred))
    #
    #                 pbar.update(1)
    #                 count += 1
    #                 if count >= data_n:
    #                     break
    #         except (StopIteration, RuntimeError):
    #             print('stop iteration ...')
    #
    #     df = pd.DataFrame(report, index=indxs)
    #     return df

    # def _batch_eval_report(self, data_gen, n_iter, model):
    #     #TODO: write this method for performance improvements
    #     pass

    @abstractmethod
    def get_eval_funcs(self) -> EvalFuncs:
        """Evaluation functions to use for evaluation.

        You should take ``model.predict`` outputs into account. Your functions must take ``(y_true, y_pred)`` as
        inputs (not batch), and return a single output for this data-point.

        Returns:
            a dictionary mapping from metric names to metric functions: ``{'iou': iou_metric, 'dice': get_dice_metric(), ...}``

        """
