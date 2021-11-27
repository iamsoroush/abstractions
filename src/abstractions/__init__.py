from pathlib import Path

MLFLOW_TRACKING_URI = Path('/home').joinpath('vafaeisa').joinpath('mlruns')
scratch_drive = Path('/home').joinpath('vafaeisa').joinpath('scratch')
EVAL_REPORTS_DIR = scratch_drive.joinpath('eval_reports')
PROJECT_ROOT = Path(__file__).parent.parent

from .base_class import BaseClass
from .model_building import ModelBuilderBase
from .data_loading import DataLoaderBase
from .evaluation import EvaluatorBase
from .preprocessing import PreprocessorBase
from .augmentation import AugmentorBase
from .training import Trainer
from .orchestration import Orchestrator
