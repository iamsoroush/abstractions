import yaml
import pathlib
import logging
import sys

import mlflow

from .abs_exceptions import ConfigNotFound, FoundMultipleConfigs


def setup_mlflow(mlflow_tracking_uri, mlflow_experiment_name: str,
                 base_dir: pathlib.Path, evaluation=False) -> mlflow.ActiveRun:
    """Sets up mlflow and returns an ``active_run`` object.

    tracking_uri/
        experiment_id/
            run1
            run2
            ...

    Args:
        mlflow_tracking_uri: ``tracking_uri`` for mlflow
        mlflow_experiment_name: ``experiment_name`` for mlflow, use the same ``experiment_name`` for all experiments
        related to the same task. This is different from the ``experiment`` concept that we use.
        base_dir: directory for your experiment, containing your `config.yaml` file.
        evaluation: if evaluation==true, then new run will be created, named ``base_dir.name + _evaluation``

    Returns:
        active_run: an ``active_run`` object to use for mlflow logging.

    """

    # Loads run_id if exists
    run_id = None
    run_id_path = base_dir.joinpath('run_id.txt')
    run_name = base_dir.name
    nested = False
    if evaluation:
        run_name += '_evaluation'
        nested = True
    elif run_id_path.exists():
        with open(run_id_path, 'r') as f:
            run_id = f.readline()

    # mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient(mlflow_tracking_uri)

    # Create new run if run_id does not exist
    if run_id is not None:
        mlflow.set_experiment(mlflow_experiment_name)
        active_run = mlflow.start_run(run_id=run_id)
    else:
        experiment = client.get_experiment_by_name(mlflow_experiment_name)
        if experiment is not None:
            experiment_id = experiment.experiment_id
        else:
            experiment_id = mlflow.create_experiment(mlflow_experiment_name)

        active_run = mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=nested)

    return active_run


def check_for_config_file(run_dir: pathlib.Path) -> pathlib.Path:
    """Checks for existence of config file and returns the path to config file if exists.

    Raises:
        ConfigNotFound
        FoundMultipleConfigs

    Returns:
        path to config file

    """

    if not run_dir.is_dir():
        raise Exception(f'{run_dir} is not a directory.')

    yaml_files = list(run_dir.glob('*.yaml'))
    if not any(yaml_files):
        raise ConfigNotFound(f'no .yaml files found.')
    elif len(yaml_files) > 1:
        raise FoundMultipleConfigs(f'found more than one .yaml files.')

    return yaml_files[0]


class ConfigStruct:
    """Structure for loading config as a Python object.

    Attributes:
        seed (int):
        input_height (int): for resizing and model creation, this is input-height of your model's input and preprocessor's output
        input_width (int): for resizing and model creation, this is input-width of your model's input and preprocessor's output
        src_code_path (str): relative to project(repository)_dir,
        data_dir (str): required for testing, provide absolute path to dataset, your data-loader should work using this path. you can provide a different dataset directory when submitting a training job.
        data_loader_class (str): Required, relative to `src_code_path`
        model_builder_class (str): Required, relative to `src_code_path`
        preprocessor_class (str): Required, relative to `src_code_path`
        augmentor_class (str): relative to `src_code_path`
        evaluator_class (str): Required, relative to `src_code_path`
        epochs (int):
        batch_size (int):
        data_loader (Struct): parameters for instantiating DataLoader
        model_builder (Struct): parameters for instantiating ModelBuilder
        preprocessor (Struct): parameters for instantiating Preprocessor
        augmentor (Struct): parameters for instantiating Augmentor
        do_train_augmentation (bool):
        do_validation_augmentation (bool):
        export (Struct): parameters for exporting, will be used by trainer
        project_name (str):

    """

    def __init__(self, **entries):
        self.seed = 101
        self.input_height = None
        self.input_width = None
        self.src_code_path = None
        self.data_dir = None
        self.data_loader_class = None
        self.model_builder_class = None
        self.preprocessor_class = None
        self.augmentor_class = None
        self.evaluator_class = None
        self.epochs = None
        self.batch_size = None
        self.data_loader = None
        self.model_builder = None
        self.preprocessor = None
        self.augmentor = None
        self.do_train_augmentation = None
        self.do_validation_augmentation = None
        self.export = Struct(metric='val_loss', mode='min')
        self.project_name = None

        for k, v in entries.items():
            if isinstance(v, dict):
                self.__dict__[k] = Struct(**v)
            else:
                self.__dict__[k] = v


class Struct:
    def __init__(self, **entries):
        for k, v in entries.items():
            if isinstance(v, dict):
                self.__dict__[k] = Struct(**v)
            else:
                self.__dict__[k] = v


def load_config_file(path: pathlib.Path) -> ConfigStruct:
    """
    loads the json config file and returns a dictionary

    Args:
        path: path to json config file

    Returns:
        a nested object in which parameters are accessible using dot notations, for example ``config.model.optimizer.lr``

    """

    with open(path) as f:
        data_map = yaml.safe_load(f)

    config_obj = ConfigStruct(**data_map)
    return config_obj


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # if write_logs:
    #     dir_path = os.path.dirname(os.path.realpath(__file__))
    #     log_dir = os.path.join(dir_path, 'logs')
    #     if not os.path.exists(log_dir):
    #         os.mkdir(log_dir)
    #     file_handler = logging.FileHandler(os.path.join(log_dir, "{}.log".format(name)))
    #     file_handler.setFormatter(formatter)
    #     logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    return logger
