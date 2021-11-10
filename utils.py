import yaml
import pathlib

import mlflow
from mlflow.tracking import MlflowClient


def setup_mlflow(mlflow_tracking_uri, mlflow_experiment_name, base_dir: pathlib.Path):
    """Sets up mlflow and returns an ``active_run`` object.

    tracking_uri/
        experiment_id/
            run1
            run2
            ...

    Attributes:
        mlflow_tracking_uri: ``tracking_uri`` for mlflow
        mlflow_experiment_name: ``experiment_name`` for mlflow, use the same ``experiment_name`` for all experiments
        related to the same task. This is different from the ``experiment`` concept that we use.
        base_dir: directory for your experiment, containing your `config.yaml` file.

    Returns:
        active_run: an ``active_run`` object to use for mlflow logging.

    """

    # Loads run_id if exists
    run_id_path = base_dir.joinpath('run_id.txt')
    run_name = base_dir.name

    if run_id_path.exists():
        with open(run_id_path, 'r') as f:
            run_id = f.readline()
    else:
        run_id = None

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient(mlflow_tracking_uri)

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

        active_run = mlflow.start_run(experiment_id=experiment_id, run_name=run_name)

    return active_run


def check_for_config_file(run_dir: pathlib.Path):
    """Checks for existence of config file and returns the path to config file if exists."""

    if not run_dir.is_dir():
        raise Exception(f'{run_dir} is not a directory.')

    yaml_files = list(run_dir.glob('*.yaml'))
    if not any(yaml_files):
        raise Exception(f'no .yaml files found.')
    elif len(yaml_files) > 1:
        raise Exception(f'found more than one .yaml files.')

    return yaml_files[0]


def load_config_file(path: pathlib.Path):
    """
    loads the json config file and returns a dictionary

    Attributes:
        path: path to json config file

    Returns:
        a nested object in which parameters are accessible using dot notations for example ``config.model.optimizer.lr``

    """

    with open(path) as f:
        data_map = yaml.safe_load(f)

    config_obj = Struct(**data_map)
    return config_obj


class Struct:
    def __init__(self, **entries):
        for k, v in entries.items():
            if isinstance(v, dict):
                self.__dict__[k] = Struct(**v)
            else:
                self.__dict__[k] = v
