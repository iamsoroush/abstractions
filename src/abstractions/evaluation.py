from abc import abstractmethod, ABC

import mlflow
from tqdm import tqdm
import typing

import numpy as np
import pandas as pd
import tensorflow as tf

from .base_class import BaseClass
from .model_building import MBBase
from .data_loading import DataLoaderBase
from .preprocessing import PreprocessorBase
from aimedeval import EvalFunc


# EvalFuncs = typing.Dict[str, typing.Callable]


class EvalBase(BaseClass):

    @abstractmethod
    def evaluate(self,
                 data_loader: DataLoaderBase,
                 preprocessor: PreprocessorBase,
                 model: MBBase,
                 active_run: mlflow.ActiveRun) -> pd.DataFrame:
        """Evaluate on evaluation(test) set."""

    @abstractmethod
    def validation_evaluate(self,
                            data_loader: DataLoaderBase,
                            preprocessor: PreprocessorBase,
                            exported_model: MBBase,
                            active_run: typing.Optional[mlflow.ActiveRun],
                            index) -> pd.DataFrame:
        """Evaluate on validation set."""

    @abstractmethod
    def get_eval_funcs(self) -> typing.List[EvalFunc]:
        """Evaluation functions to use for evaluation.

        You should take ``model.predict`` outputs into account. Your functions must take ``(y_true, y_pred)`` as
        inputs (not batch), and return a single output for this data-point.

        Returns:
            a list of ``EvalFunc``s``

        """


class EvaluatorBase(EvalBase, ABC):

    def _load_params(self, config):
        pass

    def _set_defaults(self):
        pass

    def evaluate(self,
                 data_loader: DataLoaderBase,
                 preprocessor: PreprocessorBase,
                 model: MBBase,
                 active_run: mlflow.ActiveRun) -> pd.DataFrame:
        """Evaluates the model using ``eval_functions`` defined in ``get_eval_functions`` on test dataset.

        Notes:
            - ``create_test_generator`` method of the ``data_loader`` will be used to create a data generator, and it is expected to have a tuple of ``(image, label, data_id)`` as output.

        Args:
            data_loader: will be used for creating test-data-generator
            preprocessor: will be used to add image-preprocess to the test-data-generator
            model: a ``ModelBuilder`` or ``GenericModelBuilder`` that provides ``.evaluate`` and ``.predict`` method.
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
        data_gen = iter(wrapper_gen(iter(test_data_gen)))
        for k, v in model._evaluate(data_gen, n_iter_test).items():
            eval_internal_metrics[f'_model.evaluate_{k}'] = v
        self._log_metrics_to_mlflow(active_run, eval_internal_metrics, prefix='test')

        # Eval funcs
        test_data_gen, test_n = data_loader.create_test_generator()
        test_data_gen = preprocessor.add_image_preprocess(test_data_gen)
        test_data_gen = preprocessor.add_label_preprocess(test_data_gen)
        test_data_gen, n_iter_test = preprocessor.batchify(test_data_gen, test_n)

        preds, gts, data_ids = model._predict(test_data_gen, n_iter_test)
        report_df = self._generate_eval_reports(preds, gts, data_ids)
        self._log_df_report_to_mlflow(active_run, report_df, prefix='test')

        return report_df

    def validation_evaluate(self,
                            data_loader: DataLoaderBase,
                            preprocessor: PreprocessorBase,
                            model: MBBase,
                            active_run: typing.Optional[mlflow.ActiveRun],
                            index) -> pd.DataFrame:
        """Evaluates the model using ``eval_functions`` defined in ``get_eval_functions`` on validation dataset.

        Notes:
            - ``create_validation_generator`` method of the ``data_loader`` will be used to create a data generator which yields a tuple of ``(image, label, sample_weight)`` as output.

        Args:
            data_loader: will be used for creating test-data-generator.
            preprocessor: will be used to add image-preprocess to the test-data-generator.
            model: a ``ModelBuilder`` or ``GenericModelBuilder`` that provides ``.evaluate`` and ``.predict`` method.
            active_run: mlflow's ``ActiveRun`` instance to log evaluation reports to.
            index: a list of ``data_id`` s to use as report data-frame's index. use this if your validation generator does not return a unique ``data_id`` as third element.

        Returns:
            a ``pandas.DataFrame`` with index= ``data_id`` s, each row represents the result for a single data-point
            and each column represents a metric.

        """

        def wrapper_gen_dot_evaluate(gen):
            while True:
                x_b, y_b, w_b = next(gen)
                yield x_b, y_b

        def wrapper_gen_dot_predict(gen, index):
            ind = 0
            while True:
                x_b, y_b, w_b = next(gen)
                n_batch = len(x_b)
                data_ids = index[ind: ind + n_batch]
                ind += n_batch
                yield x_b, y_b, data_ids

        # Internal eval metrics + loss value
        val_data_gen, val_n = data_loader.create_validation_generator()
        val_data_gen, n_iter_val = preprocessor.add_preprocess(val_data_gen, n_data_points=val_n)

        eval_internal_metrics = dict()
        data_gen = iter(wrapper_gen_dot_evaluate(iter(val_data_gen)))
        for k, v in model._evaluate(data_gen, n_iter_val).items():
            eval_internal_metrics[f'_model.evaluate_{k}'] = v
        self._log_metrics_to_mlflow(active_run, eval_internal_metrics, prefix='validation')

        # Eval funcs
        val_data_gen, val_n = data_loader.create_validation_generator()
        val_data_gen, n_iter_val = preprocessor.add_preprocess(val_data_gen, n_data_points=val_n)
        if index is None:
            index = np.array([[i] for i in range(val_n)])
        elif np.array(index).ndim != 2:
            index = np.array([[i] for i in index])

        data_gen = iter(wrapper_gen_dot_predict(iter(val_data_gen), index))
        preds, gts, data_ids = model._predict(data_gen, n_iter_val)
        report_df = self._generate_eval_reports(preds, gts, data_ids)
        self._log_df_report_to_mlflow(active_run, report_df, prefix='validation')

        return report_df

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

    def _generate_eval_reports(self, preds, gts, data_ids):
        eval_funcs = self.get_eval_funcs()
        eval_func_dict = {func.name: func.get_func() for func in eval_funcs}
        report = {func.name: list() for func in eval_funcs}
        # report = {k: list() for k in eval_funcs.keys()}
        n_data = len(preds)
        indxs = list()
        with tqdm(total=n_data) as pbar:
            for ind, (y_pred, y_true, data_id) in enumerate(zip(preds, gts, data_ids)):
                for k, v in report.items():
                    metric_val = eval_func_dict[k](y_true, y_pred)
                    if isinstance(metric_val, tf.Tensor):
                        v.append(metric_val.numpy())
                    else:
                        v.append(metric_val)

                if np.array(data_id).ndim > 0:
                    d_id = data_id[0]
                else:
                    d_id = data_id
                if isinstance(d_id, tf.Tensor):
                    d_id = d_id.numpy()
                    if isinstance(d_id, bytes):
                        indxs.append(d_id.decode())
                    else:
                        indxs.append(d_id)
                else:
                    indxs.append(str(d_id))

                # if hasattr(data_id, '__iter__'):
                #     indxs.append(data_id[0])
                # else:
                # if isinstance(data_id, tf.Tensor):
                #     indxs.append(data_id.numpy())
                # else:
                #     indxs.append(data_id)
                pbar.update(1)

        df = pd.DataFrame(report, index=indxs)
        return df
