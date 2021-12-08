import pytest
from pathlib import Path
from abstractions.utils import load_config_file, Struct


# @pytest.fixture()
# def run_config(pytestconfig):
#     run_name = pytestconfig.getoption("run_name")
#     run_dir = Path('runs').joinpath(run_name)
#     config = load_config_file(list(run_dir.glob('*.yaml'))[0])
#     return config

def get_configs():
    config_files = list()
    for config_path in Path('runs').rglob('*.yaml'):
        config_files.append(config_path)
    return config_files


@pytest.mark.init
def test_requirements():
    """Tests for main requirements."""

    packages = [
        "tensorflow>=2.7.0",
        "pandas",
        "scikit-learn",
        "scikit-image",
        "scipy",
        "matplotlib",
        "mlflow",
        "abstractions-aimedic",
        "SimpleITK",
        "pyyaml",
        "albumentations",
        "tqdm",
        "pytest",
        "pytest-dependency"
    ]

    # project_dir = Path(__file__).parent.parent
    # print(project_dir)

    # with open(project_dir.joinpath("requirements.txt"), "r") as f:
    with open('requirements.txt', 'r') as f:
        reqs = f.read().split()
        assert all(item in reqs for item in packages)


@pytest.mark.init
@pytest.mark.parametrize('config_path', get_configs())
def test_required_config_fields(config_path):
    """Testing ``required`` config parameters"""

    run_config = load_config_file(config_path)

    assert str(run_config.input_height).isalnum()
    assert str(run_config.input_width).isalnum()
    assert isinstance(run_config.src_code_path, str)
    assert isinstance(run_config.data_dir, str)
    assert isinstance(run_config.data_loader_class, str)
    assert isinstance(run_config.preprocessor_class, str)
    assert isinstance(run_config.model_builder_class, str)
    assert isinstance(run_config.evaluator_class, str)

    assert isinstance(run_config.export, Struct)
    assert hasattr(run_config.export, 'metric')
    assert hasattr(run_config.export, 'mode')


@pytest.mark.init
def test_train_script_existence():
    """Testing whether train script exists in ``scripts`` or not"""

    train_path = Path('scripts').joinpath('train.py')
    assert train_path.is_file()
