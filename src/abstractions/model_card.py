import typing
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import model_card_toolkit as mctlib
from model_card_toolkit import ModelCardToolkit
from model_card_toolkit.utils.graphics import figure_to_base64str
from abstractions.utils import ConfigStruct


class ModelCardGenerator:

    def __init__(self, model_card_dir: Path):
        self.model_card_dir = model_card_dir
        self.model_card_dir.mkdir(exist_ok=True)

        self.mct = ModelCardToolkit(str(self.model_card_dir))
        self.model_card = self.mct.scaffold_assets()

    def generate(self,
                 config: ConfigStruct,
                 val_eval_report: pd.DataFrame = None,
                 eval_eval_report: pd.DataFrame = None) -> typing.Tuple[str, str]:
        self._add_model_details(config)
        self._add_model_parameters(config)
        self._add_considerations(config)
        self._add_qa(val_eval_report, eval_eval_report)

        self.mct.update_model_card(self.model_card)
        html_doc = self.mct.export_format(self.model_card)
        md_path = self.model_card_dir.joinpath('template/md/default_template.md.jinja')
        md_doc = self.mct.export_format(template_path=str(md_path), output_file='../../README.md')
        return html_doc, md_doc

    def _add_model_details(self, config: ConfigStruct):
        # Model Details
        try:
            model_details = config.model_details
        except AttributeError:
            pass
        else:
            try:
                name = model_details.name
                self.model_card.model_details.name = name
            except AttributeError:
                pass
            try:
                overview = model_details.overview
                self.model_card.model_details.overview = overview
            except AttributeError:
                pass
            try:
                doc = model_details.documentation
                self.model_card.model_details.documentation = doc
            except AttributeError:
                pass

    def _add_model_parameters(self, config: ConfigStruct):
        # Model Parameters
        try:
            model_parameters = config.model_parameters
        except AttributeError:
            pass
        else:
            try:
                arch = model_parameters.model_architecture
                self.model_card.model_parameters.model_architecture = arch
            except AttributeError:
                pass
            try:
                data = model_parameters.data
                self.model_card.model_parameters.data = [mctlib.Dataset(name=data.name,
                                                                        description=data.description,
                                                                        link=data.link)]
            except AttributeError:
                pass
            try:
                input_format = model_parameters.input_format
                self.model_card.model_parameters.input_format = input_format
            except AttributeError:
                pass
            try:
                output_format = model_parameters.output_format
                self.model_card.model_parameters.output_format = output_format
            except AttributeError:
                pass

    def _add_considerations(self, config: ConfigStruct):
        # Considerations
        try:
            considerations = config.considerations
        except AttributeError:
            pass
        else:
            try:
                users = considerations.users
                self.model_card.considerations.users = [mctlib.User(description=user) for user in users]
            except AttributeError:
                pass
            try:
                use_cases = considerations.use_cases
                self.model_card.considerations.use_cases = [mctlib.UseCase(description=use_case) for use_case in
                                                            use_cases]
            except AttributeError:
                pass
            try:
                limitations = considerations.limitations
                self.model_card.considerations.limitations = [mctlib.Limitation(description=lim) for lim in limitations]
            except AttributeError:
                pass

    def _add_qa(self, val_df: pd.DataFrame, eval_df: pd.DataFrame):
        # Quantitative analysis
        performance_metrics = list()
        graphics = list()

        if val_df is not None:
            val_eval_report_summary = val_df.describe()
            val_metric_names = val_eval_report_summary.columns
            for metric in val_metric_names:
                pm = mctlib.PerformanceMetric(type=metric,
                                              value='{:.3f}'.format(val_eval_report_summary[metric]['mean']),
                                              slice='validation')
                performance_metrics.append(pm)

                fig = self._plot_metrics(val_df, val_metric_names)
            # fig, axes = plt.subplots(nrows=len(val_metric_names), ncols=1,
            #                          figsize=(8, 4 * len(eval_metric_names)))
            # for i, col in enumerate(val_metric_names):
            #     ax = axes[i]
            #     ax.hist(val_df[col], bins=50)
            #     ax.set_title(f'Distribution of {col}')
            val_graphic = mctlib.Graphic(name='Eval Func distributions on validation data',
                                         image=figure_to_base64str(fig))
            graphics.append(val_graphic)

        if eval_df is not None:
            eval_eval_report_summary = eval_df.describe()
            eval_metric_names = eval_eval_report_summary.columns
            for metric in eval_metric_names:
                pm = mctlib.PerformanceMetric(type=metric,
                                              value='{:.3f}'.format(eval_eval_report_summary[metric]['mean']),
                                              slice='evaluation')
                performance_metrics.append(pm)

                fig = self._plot_metrics(eval_df, eval_metric_names)
            # fig, axes = plt.subplots(nrows=len(eval_metric_names), ncols=1,
            #                          figsize=(8, 4 * len(eval_metric_names)))
            # for i, col in enumerate(eval_metric_names):
            #     ax = axes[i]
            #     ax.hist(eval_df[col], bins=50)
            #     ax.set_title(f'Distribution of {col}')
            eval_graphic = mctlib.Graphic(name='Eval Func distributions on evaluation data',
                                          image=figure_to_base64str(fig))
            graphics.append(eval_graphic)

        self.model_card.quantitative_analysis.performance_metrics = performance_metrics
        self.model_card.quantitative_analysis.graphics = mctlib.GraphicsCollection(description='EvalFunc distributions',
                                                                                   collection=graphics)

    def _plot_metrics(self, df, metric_names):
        fig, axes = plt.subplots(nrows=len(metric_names), ncols=1,
                                 figsize=(8, 4 * len(metric_names)))
        for i, col in enumerate(metric_names):
            ax = axes[i]
            ax.hist(df[col], bins=50)
            ax.set_title(f'Distribution of {col}')

        return fig
