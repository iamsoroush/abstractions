import yaml
import pathlib
import logging
import sys
import mlflow
from .abs_exceptions import ConfigNotFound, FoundMultipleConfigs
from mlflow.exceptions import MlflowException


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
            try:
                experiment_id = mlflow.create_experiment(mlflow_experiment_name)
            except MlflowException as e:
                logging.warning(f"mlflow-experiment is not found and can't be created, {e} setting experiment id to 0")
                experiment_id = 0

        active_run = mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=nested)

    return active_run


def add_config_file_to_mlflow(config_dict: dict):
    """Adds parameters from config file to mlflow.

    Args:
        config_dict: config file as a nested dictionary
    """

    def param_extractor(dictionary):

        """Returns a list of each item formatted like 'trainer.mlflow.tracking_uri: /tracking/uri' """

        values = []
        if dictionary is None:
            return values

        for key, value in dictionary.items():
            if isinstance(value, dict):
                items_list = param_extractor(value)
                for i in items_list:
                    values.append(f'{key}.{i}')
            else:
                values.append(f'{key}: {value}')
        return values

    fields_to_ignore = ['model_details', 'model_parameters', 'considerations']
    new_config = {k: v for k, v in config_dict.items() if k not in fields_to_ignore}
    str_params = param_extractor(new_config)
    params = {}
    for item in str_params:
        name = f"config_{item.split(':')[0]}"
        item_value = item.split(': ')[-1]

        params[name] = item_value

    mlflow.log_params(params)


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
        self.input_channels = None
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
        self.model_details = Struct(name=None, overview=None, documentation=None)
        self.model_parameters = Struct(model_architecture=None,
                                       data=Struct(name=None, description=None, link=None),
                                       input_format=None,
                                       output_format=None)
        self.considerations = Struct(users=list(), use_cases=list(), limitations=list())

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
    loads the ``yaml`` config file and returns a ``ConfigStruct``

    Args:
        path: path to json config file

    Returns:
        a nested object in which parameters are accessible using dot notations, for example ``config.model.optimizer.lr``

    """

    config_obj = ConfigStruct(**load_config_as_dict(path))
    return config_obj


def load_config_as_dict(path: pathlib.Path) -> dict:
    """
    loads the ``yaml`` config file and returns a dictionary

    Args:
        path: path to json config file

    Returns:
        a nested object in which parameters are accessible using dot notations, for example ``config.model.optimizer.lr``

    """

    with open(path) as f:
        data_map = yaml.safe_load(f)
    return data_map


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
