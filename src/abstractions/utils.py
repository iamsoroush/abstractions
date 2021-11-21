import yaml
import pathlib
import logging
import sys

from .abs_exceptions import ConfigNotFound, FoundMultipleConfigs


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
        input_height (int):
        input_width (int):
        src_code_path (str):
        data_loader_class (str):
        model_builder_class (str):
        preprocessor_class (str):
        augmentor_class (str):
        evaluator_class (str):
        epochs (int):
        batch_size (int):
        data_loader (Struct):
        model_builder (Struct):
        preprocessor (Struct):
        augmentor (Struct):
        do_train_augmentation (bool):
        do_validation_augmentation (bool):
        export (Struct):
        project_name (str):

    """

    def __init__(self, **entries):
        self.seed = 101
        self.input_height = None
        self.input_width = None
        self.src_code_path = None
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
        self.export = None
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
