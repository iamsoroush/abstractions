"""
pydoc.locate locates the class and creates an instance of it.
run_config: This will run the run_config file to get the config from config.yaml defined by users.
By using this variable, one can access variables in the config file.
component_holder: For testing different modules several components are needed.
This variable stores outputs of previous tests up to that specific test to make upcoming tests possible.
To be aware of the contents of component_holder, follow the dependencies of the current testing module.
"""
import os.path
import warnings
import pytest
from pathlib import Path
from pydoc import locate
import tensorflow as tf
from abstractions.utils import load_config_file
from abstractions import DataLoaderBase, AugmentorBase, PreprocessorBase, ModelBuilderBase, EvaluatorBase
from deep_utils import color_str

warnings.filterwarnings('ignore')


@pytest.fixture(scope='module')
def component_holder():
    return {}


@pytest.fixture()
def run_config(pytestconfig):
    config_path = pytestconfig.getoption('config_path')
    if config_path is None:
        run_name = pytestconfig.getoption('run_name')
        run_dir = Path('runs').joinpath(run_name)
        yaml_files = list(run_dir.glob('*.yaml'))
        if len(yaml_files) == 0:
            raise ValueError(f"[ERROR] run-dir {run_dir} has not config.yaml file")
        config = load_config_file(yaml_files[0])
    elif os.path.isfile(config_path):
        config = load_config_file(config_path)
    else:
        raise ValueError(f"[ERROR] config-path {config_path} does not exist")
    return config


@pytest.mark.dependency
@pytest.mark.dataloader
class TestDataLoader:

    def test_locate_data_loader_class(self, run_config, component_holder):
        """Whether package can locate data_loader_class. The relative address for data_loader_class is set in config.yaml"""

        data_loader_class = locate(run_config.data_loader_class)
        assert data_loader_class is not None
        component_holder['data_loader_class'] = data_loader_class

    @pytest.mark.dependency(depends=['TestDataLoader::test_locate_data_loader_class'])
    def test_initialize_data_loader(self, run_config, component_holder):
        """if data_loader could be initialized correctly."""

        # data_dir = Path(run_config.data_dir)
        # data_loader = component_holder.get('data_loader_class')(run_config, data_dir)
        data_loader = component_holder.get('data_loader_class')(run_config)
        assert isinstance(data_loader, DataLoaderBase)
        component_holder['data_loader'] = data_loader

    # ---------- Train Data ----------
    @pytest.mark.dependency(depends=['TestDataLoader::test_initialize_data_loader'])
    def test_create_training_generator(self, run_config, component_holder):
        data_loader = component_holder['data_loader']
        ret = data_loader.create_training_generator()
        assert len(ret) == 2, "Provide a generator and the number of training samples"

        train_data_gen, train_n = ret
        component_holder['train_data_gen'] = train_data_gen
        component_holder['train_n'] = train_n

    @pytest.mark.dependency(depends=['TestDataLoader::test_create_training_generator'])
    def test_train_n(self, run_config, component_holder):
        assert hasattr(component_holder['train_data_gen'], '__iter__')

    @pytest.mark.dependency(depends=['TestDataLoader::test_create_training_generator'])
    def test_train_generator(self, run_config, component_holder):
        assert isinstance(component_holder['train_n'], int)

    # @pytest.mark.dependency(depends=['TestDataLoader::test_train_generator'])
    # def test_train_gen_out(self, component_holder):
    #     """Checks train generator's number of outputs"""
    #     gen = component_holder['train_data_gen']
    #     ret = next(iter(gen))
    #     assert len(
    #         ret) == 3, color_str(
    #         f"Generator returns two components. Ignore this message if you don't have weights for your samples",
    #         "yellow", mode="bold") if len(
    #         ret) == 2 else color_str("Generator does not return a proper number of outputs!", "red",
    #                                  mode=["bold", "underline"])

    @pytest.mark.dependency(depends=['TestDataLoader::test_train_generator'])
    def test_train_gen_out(self, run_config, component_holder):
        train_gen = component_holder['train_data_gen']
        ret = next(iter(train_gen))
        assert 2 <= len(ret) < 4, color_str(f"Generator does not return a proper number of outputs, {len(ret)}!", "red",
                                            mode=["bold", "underline"])

    # ---------- Validation Data ----------
    @pytest.mark.dependency(depends=['TestDataLoader::test_initialize_data_loader'])
    def test_create_validation_generator(self, run_config, component_holder):
        data_loader = component_holder['data_loader']
        ret = data_loader.create_validation_generator()

        assert len(ret) == 2, color_str(
            f"test_create_validation_generator should return a generator and dataset count", "red",
            mode=["bold", "underline"])
        validation_data_gen, validation_n = ret
        component_holder['validation_data_gen'] = validation_data_gen
        component_holder['validation_n'] = validation_n

    @pytest.mark.dependency(depends=['TestDataLoader::test_create_validation_generator'])
    def test_validation_n(self, run_config, component_holder):
        assert hasattr(component_holder['validation_data_gen'], '__iter__')

    @pytest.mark.dependency(depends=['TestDataLoader::test_create_validation_generator'])
    def test_validation_generator(self, run_config, component_holder):
        assert isinstance(component_holder['validation_n'], int)

    @pytest.mark.dependency(depends=['TestDataLoader::test_create_validation_generator'])
    def test_validation_gen_out(self, component_holder):
        gen = component_holder['validation_data_gen']
        ret = next(iter(gen))
        assert 2 <= len(ret) < 4, color_str(f"Generator does not return a proper number of outputs, {len(ret)}!", "red",
                                            mode=["bold", "underline"])

    # ---------- Test Data ----------
    @pytest.mark.dependency(depends=['TestDataLoader::test_initialize_data_loader'])
    def test_create_evaluation_generator(self, run_config, component_holder):
        """Checking evaluation/test dataset"""
        try:
            data_loader = component_holder['data_loader']
            ret = data_loader.create_test_generator()
            assert len(ret) == 2, color_str(
                "test_create_validation_generator should return a generator and dataset count", "red",
                mode=["bold", "underline"])
        except Exception as e:
            raise Exception(color_str(
                f"Error in evaluation generation, {e}", "red",
                mode=["bold", "underline"]))
        evaluation_data_gen, evaluation_n = ret
        component_holder['evaluation_data_gen'] = evaluation_data_gen
        component_holder['evaluation_n'] = evaluation_n

    @pytest.mark.dependency(depends=['TestDataLoader::test_create_evaluation_generator'])
    def test_evaluation_n(self, run_config, component_holder):
        assert hasattr(component_holder['evaluation_data_gen'], '__iter__')

    @pytest.mark.dependency(depends=['TestDataLoader::test_create_evaluation_generator'])
    def test_evaluation_generator(self, run_config, component_holder):
        assert isinstance(component_holder['evaluation_n'], int)

    @pytest.mark.dependency(depends=['TestDataLoader::test_create_evaluation_generator'])
    def test_evaluation_gen_out(self, component_holder):
        gen = component_holder['evaluation_data_gen']
        ret = next(iter(gen))
        assert 2 <= len(ret) < 4, color_str(f"Generator does not return a proper number of outputs, {len(ret)}!", "red",
                                            mode=["bold", "underline"])


@pytest.mark.augmentation
class TestAugmentor:
    @pytest.mark.dependency()
    def test_locate_augmentor_class(self, run_config, component_holder):
        if not hasattr(run_config, 'augmentor_class') or run_config.augmentor_class is None:
            assert True
        else:
            print(f"augmentor_class: {run_config.augmentor_class}")
            augmentor_class = locate(run_config.augmentor_class)
            assert augmentor_class is not None
            component_holder['augmentor_class'] = augmentor_class

    @pytest.mark.dependency(depends=['TestAugmentor::test_locate_augmentor_class'])
    def test_initialize_augmentor(self, run_config, component_holder):
        """if augmentor could be initialized correctly."""

        augmentor_class = component_holder.get('augmentor_class')
        if augmentor_class is not None:
            augmentor = augmentor_class(run_config)
            assert isinstance(augmentor, AugmentorBase)
            component_holder['augmentor'] = augmentor

    # ---------- Train Augmentation ----------
    @pytest.mark.dependency(depends=['TestDataLoader::test_train_generator'])
    def test_add_train_augmentation(self, run_config, component_holder):
        augmentor = component_holder.get('augmentor')
        if augmentor is None:
            assert True
        else:
            if run_config.do_train_augmentation:
                data_gen = augmentor.add_augmentation(component_holder['train_data_gen'])
                assert hasattr(data_gen, '__iter__')
                component_holder['train_data_gen'] = data_gen

    @pytest.mark.dependency(depends=['TestAugmentor::test_add_train_augmentation'])
    def test_train_augmentation_out(self, run_config, component_holder):
        try:
            train_data_gen = component_holder['train_data_gen']
            ret = next(iter(train_data_gen))
            assert 2 <= len(ret) < 4, color_str(f"Generator does not return a proper number of outputs, {len(ret)}!",
                                                "red",
                                                mode=["bold", "underline"])
        except Exception as e:
            raise Exception(color_str(
                f"Error {e}, If you don't have augmentor class ignore this message", "yellow",
                mode=["bold"]))

    # ---------- Validation Augmentation ----------
    @pytest.mark.dependency(depends=['TestDataLoader::test_validation_generator'])
    def test_add_validation_augmentation(self, run_config, component_holder):
        augmentor = component_holder.get('augmentor')
        if augmentor is None:
            assert True
        else:
            if run_config.do_validation_augmentation:
                data_gen = augmentor.add_validation_augmentation(component_holder['validation_data_gen'])
                assert hasattr(data_gen, '__iter__')
                component_holder['validation_data_gen'] = data_gen

    @pytest.mark.dependency(depends=['TestAugmentor::test_add_validation_augmentation'])
    def test_validation_augmentation_out(self, run_config, component_holder):
        try:
            data_gen = component_holder['validation_data_gen']
            ret = next(iter(data_gen))
            assert 2 <= len(ret) < 4, color_str(f"Generator does not return a proper number of outputs, {len(ret)}!",
                                                "red",
                                                mode=["bold", "underline"])
        except Exception as e:
            raise Exception(color_str(
                f"Error {e}, If you don't have augmentor class ignore this message", "yellow",
                mode=["bold"]))


# Not augmentation are added for evaluation mode

@pytest.mark.preprocessor
class TestPreprocessor:
    @pytest.mark.dependency()
    def test_locate_preprocessor_class(self, run_config, component_holder):
        """if package can locate preprocessor_class"""

        preprocessor_class = locate(run_config.preprocessor_class)
        assert preprocessor_class is not None
        component_holder['preprocessor_class'] = preprocessor_class

    @pytest.mark.dependency(depends=['TestPreprocessor::test_locate_preprocessor_class'])
    def test_initialize_preprocessor(self, run_config, component_holder):
        """if preprocessor could be initialized correctly."""

        preprocessor = component_holder.get('preprocessor_class')(run_config)
        assert isinstance(preprocessor, PreprocessorBase)
        component_holder['preprocessor'] = preprocessor

    # ---------- Train Preprocessing ----------
    @pytest.mark.dependency(depends=['TestAugmentor::test_add_train_augmentation'])
    def test_add_train_preprocessing(self, run_config, component_holder):
        preprocessor = component_holder['preprocessor']
        train_data_gen = component_holder['train_data_gen']
        train_n = component_holder['train_n']

        ret = preprocessor.add_preprocess(train_data_gen, train_n)
        assert len(ret) == 2

        train_gen, n_iter_train = ret
        component_holder['train_data_gen'] = train_gen
        component_holder['n_iter_train'] = n_iter_train

    @pytest.mark.dependency(depends=['TestPreprocessor::test_add_train_preprocessing'])
    def test_train_preprocess_generator(self, run_config, component_holder):
        assert hasattr(component_holder['train_data_gen'], '__iter__')

    @pytest.mark.dependency(depends=['TestPreprocessor::test_add_train_preprocessing'])
    def test_train_preprocess_n_iter(self, run_config, component_holder):
        assert isinstance(component_holder['n_iter_train'], int)

    @pytest.mark.dependency(depends=['TestPreprocessor::test_train_preprocess_generator'])
    def test_train_gen_out(self, run_config, component_holder):
        train_gen = component_holder['train_data_gen']
        ret = next(iter(train_gen))
        assert 2 <= len(ret) < 4, color_str("Generator does not return a proper number of outputs!", "red",
                                            mode=["bold", "underline"])

        if len(ret) == 3:
            x_batch, y_batch, w_batch = ret
            component_holder['train_gen_xb_shape'] = tuple(x_batch.shape)
            component_holder['train_gen_yb_shape'] = tuple(y_batch.shape)
            component_holder['train_gen_wb'] = w_batch
        else:
            x_batch, y_batch = ret
            component_holder['train_gen_xb_shape'] = tuple(x_batch.shape)
            component_holder['train_gen_yb_shape'] = tuple(y_batch.shape)
            component_holder['train_gen_wb'] = None

    @pytest.mark.dependency(depends=['TestPreprocessor::test_train_gen_out'])
    def test_train_gen_x_shape(self, run_config, component_holder):
        batch_size = int(run_config.batch_size)
        input_h = int(run_config.input_height)
        input_w = int(run_config.input_width)
        x_b_shape = component_holder['train_gen_xb_shape']

        assert tuple(x_b_shape)[:3] == (batch_size, input_h, input_w)
        component_holder['input_channels'] = tuple(x_b_shape)[-1]

    @pytest.mark.dependency(depends=['TestPreprocessor::test_train_gen_out'])
    def test_train_gen_y_batch_size(self, run_config, component_holder):
        assert component_holder['train_gen_yb_shape'][0] == int(run_config.batch_size)

    @pytest.mark.dependency(depends=['TestPreprocessor::test_train_gen_out'])
    def test_train_gen_w_batch_size(self, run_config, component_holder):
        try:
            w_batch = component_holder['train_gen_wb']
            if w_batch is not None:
                assert len(w_batch) == int(run_config.batch_size)
        except Exception as e:
            raise Exception(color_str(
                f"Error {e}, If you don't have weights for your samples, ignore this message", "yellow",
                mode=["bold"]))

    @pytest.mark.dependency(depends=['TestPreprocessor::test_train_gen_out'])
    def test_train_gen_w_is_iterable(self, run_config, component_holder):
        try:
            w_batch = component_holder['train_gen_wb']
            if w_batch is not None:
                assert hasattr(w_batch[0], '__iter__'), color_str(
                    "Elements of the weights_batch are not iterables, add a new axis to each element.",
                    "yellow", mode=["bold"])
        except Exception as e:
            raise Exception(color_str(
                f"Error {e}, If you don't have weights for your samples, ignore this message", "yellow",
                mode=["bold"]))

    # ---------- Validation Preprocessing ----------
    @pytest.mark.dependency(depends=['TestAugmentor::test_add_validation_augmentation'])
    def test_add_validation_preprocessing(self, run_config, component_holder):
        preprocessor = component_holder['preprocessor']
        data_gen = component_holder['validation_data_gen']
        n = component_holder['validation_n']

        ret = preprocessor.add_preprocess(data_gen, n)
        assert len(ret) == 2

        gen, n_iter = ret
        component_holder['validation_data_gen'] = gen
        component_holder['n_iter_validation'] = n_iter

    @pytest.mark.dependency(depends=['TestPreprocessor::test_add_validation_preprocessing'])
    def test_validation_preprocess_generator(self, run_config, component_holder):
        assert hasattr(component_holder['validation_data_gen'], '__iter__')

    @pytest.mark.dependency(depends=['TestPreprocessor::test_add_validation_preprocessing'])
    def test_validation_preprocess_n_iter(self, run_config, component_holder):
        assert isinstance(component_holder['n_iter_validation'], int)

    @pytest.mark.dependency(depends=['TestPreprocessor::test_validation_preprocess_generator'])
    def test_validation_gen_out(self, run_config, component_holder):
        gen = component_holder['validation_data_gen']
        ret = next(iter(gen))
        assert 2 <= len(ret) < 4, color_str("Generator does not return a proper number of outputs!", "red",
                                            mode=["bold", "underline"])
        if len(ret) == 3:
            x_batch, y_batch, w_batch = ret
            component_holder['validation_gen_xb_shape'] = tuple(x_batch.shape)
            component_holder['validation_gen_yb_shape'] = tuple(y_batch.shape)
            component_holder['validation_gen_wb'] = w_batch
        else:
            x_batch, y_batch = ret
            component_holder['validation_gen_xb_shape'] = tuple(x_batch.shape)
            component_holder['validation_gen_yb_shape'] = tuple(y_batch.shape)
            component_holder['validation_gen_wb'] = None

    @pytest.mark.dependency(depends=['TestPreprocessor::test_validation_gen_out'])
    def test_validation_gen_x_shape(self, run_config, component_holder):
        batch_size = int(run_config.batch_size)
        input_h = int(run_config.input_height)
        input_w = int(run_config.input_width)
        x_b_shape = component_holder['validation_gen_xb_shape']

        assert tuple(x_b_shape)[:3] == (batch_size, input_h, input_w)
        component_holder['input_channels_validation'] = tuple(x_b_shape)[-1]

    @pytest.mark.dependency(depends=['TestPreprocessor::test_validation_gen_out'])
    def test_validation_gen_y_batch_size(self, run_config, component_holder):
        assert component_holder['validation_gen_yb_shape'][0] == int(run_config.batch_size)

    @pytest.mark.dependency(depends=['TestPreprocessor::test_validation_gen_out'])
    def test_validation_gen_w_batch_size(self, run_config, component_holder):
        try:
            w_batch = component_holder['validation_gen_wb']
            if w_batch is not None:
                assert len(w_batch) == int(run_config.batch_size)
        except Exception as e:
            raise Exception(color_str(
                f"Error {e}, If you don't have weights for your samples, ignore this message", "yellow",
                mode=["bold"]))

    @pytest.mark.dependency(depends=['TestPreprocessor::test_validation_gen_out'])
    def test_validation_gen_w_is_iterable(self, run_config, component_holder):
        try:
            w_batch = component_holder['validation_gen_wb']
            if w_batch is not None:
                assert hasattr(w_batch[0], '__iter__'), color_str(
                    "Elements of the weights_batch are not iterables, add a new axis to each element.",
                    "yellow", mode=["bold"])
        except Exception as e:
            raise Exception(color_str(
                f"Error {e}, If you don't have weights for your samples, ignore this message", "yellow",
                mode=["bold"]))

    # ---------- Evaluation Preprocessing ----------
    @pytest.mark.dependency(depends=['TestDataLoader::test_evaluation_generator'])
    def test_add_evaluation_preprocessing(self, run_config, component_holder):
        preprocessor = component_holder['preprocessor']
        data_gen = component_holder['evaluation_data_gen']
        n = component_holder['evaluation_n']

        ret = preprocessor.add_preprocess(data_gen, n)
        assert len(ret) == 2, color_str(
            f"Error, evaluation should return two objects, a data-generator and the number of iterations", "red",
            mode=["bold"])

        gen, n_iter = ret
        component_holder['evaluation_data_gen'] = gen
        component_holder['n_iter_evaluation'] = n_iter

    @pytest.mark.dependency(depends=['TestPreprocessor::test_add_evaluation_preprocessing'])
    def test_evaluation_preprocess_generator(self, run_config, component_holder):
        assert hasattr(component_holder['evaluation_data_gen'], '__iter__')

    @pytest.mark.dependency(depends=['TestPreprocessor::test_add_evaluation_preprocessing'])
    def test_evaluation_preprocess_n_iter(self, run_config, component_holder):
        assert isinstance(component_holder['n_iter_evaluation'], int)

    @pytest.mark.dependency(depends=['TestPreprocessor::test_evaluation_preprocess_generator'])
    def test_evaluation_gen_out(self, run_config, component_holder):
        gen = component_holder['evaluation_data_gen']
        ret = next(iter(gen))
        assert len(ret) == 3, color_str(
            f"Generator does not return a proper number of outputs! {len(ret)}, ids are mandetory for evaluation generator, you can pass your image paths as ids",
            "red",
            mode=["bold", "underline"])

        x_batch, y_batch, data_id = ret
        component_holder['evaluation_gen_xb_shape'] = tuple(x_batch.shape)
        component_holder['evaluation_gen_yb_shape'] = tuple(y_batch.shape)
        component_holder['evaluation_gen_id'] = data_id

    @pytest.mark.dependency(depends=['TestPreprocessor::test_evaluation_gen_out'])
    def test_evaluation_gen_id_is_iterable(self, run_config, component_holder):
        try:
            w_batch = component_holder['evaluation_gen_id']
            assert hasattr(w_batch[0], '__iter__'), color_str(
                "Elements of the weights_batch are not iterables, add a new axis to each element.",
                "yellow", mode=["bold"])
        except Exception as e:
            raise Exception(color_str(
                f"Error {e}, If you don't have weights for your samples, ignore this message", "yellow",
                mode=["bold"]))

    @pytest.mark.dependency(depends=['TestPreprocessor::test_evaluation_gen_out'])
    def test_evaluation_gen_x_shape(self, run_config, component_holder):
        batch_size = int(run_config.batch_size)
        input_h = int(run_config.input_height)
        input_w = int(run_config.input_width)
        x_b_shape = component_holder['evaluation_gen_xb_shape']

        assert tuple(x_b_shape)[:3] == (batch_size, input_h, input_w)
        component_holder['input_channels_evaluation'] = tuple(x_b_shape)[-1]

    @pytest.mark.dependency(depends=['TestPreprocessor::test_evaluation_gen_out'])
    def test_evaluation_gen_y_batch_size(self, run_config, component_holder):
        assert component_holder['evaluation_gen_yb_shape'][0] == int(run_config.batch_size)

    @pytest.mark.dependency(depends=['TestPreprocessor::test_evaluation_gen_out'])
    def test_evaluation_gen_id_batch_size(self, run_config, component_holder):
        try:
            id_batch = component_holder['evaluation_gen_id']
            assert len(id_batch) == int(run_config.batch_size)
        except Exception as e:
            raise Exception(color_str(
                f"Error {e}, If you don't have weights for your samples, ignore this message", "yellow",
                mode=["bold"]))


@pytest.mark.model
class TestModel:
    @pytest.mark.dependency()
    def test_locate_model_builder_class(self, run_config, component_holder):
        """Whether package can locate model_builder_class"""

        model_builder_class = locate(run_config.model_builder_class)
        assert model_builder_class is not None
        component_holder['model_builder_class'] = model_builder_class

    @pytest.mark.dependency(depends=['TestModel::test_locate_model_builder_class'])
    def test_initialize_model_builder(self, run_config, component_holder):
        """Whether model_builder could be initialized correctly."""

        model_builder = component_holder.get('model_builder_class')(run_config)
        assert isinstance(model_builder, ModelBuilderBase)
        component_holder['model_builder'] = model_builder

    @pytest.mark.dependency(depends=['TestPreprocessor::test_train_preprocess_generator'])
    def test_get_compiled_model(self, run_config, component_holder):
        model_builder = component_holder['model_builder']
        try:
            compiled_model = model_builder.get_compiled_model()
        except Exception as e:
            raise Exception(color_str(
                f"Error {e}, There is an in error in compilation process!", "red",
                mode=["bold", "underline"]))
        assert isinstance(compiled_model, tf.keras.models.Model), color_str(
            f"Model is not a tensorflow.keras.models.Model type", "red",
            mode=["bold", "underline"])

        component_holder['compiled_model'] = compiled_model

    @pytest.mark.dependency(depends=['TestModel::test_get_compiled_model',
                                     'TestPreprocessor::test_train_preprocess_generator'])
    def test_model_train_gen_compatibility(self, run_config, component_holder):
        compiled_model = component_holder['compiled_model']

        train_gen = component_holder['train_data_gen']
        b = next(iter(train_gen))
        if len(b) == 2:
            x_b, y_b = b
            compiled_model.evaluate(x=x_b, y=y_b)
        elif len(b) == 3:
            x_b, y_b, w_b = b
            compiled_model.evaluate(x=x_b, y=y_b, sample_weight=w_b)
        else:
            raise ValueError(color_str(
                f"Generator does not return a proper number of outputs", "red",
                mode=["bold", "underline"]))

    @pytest.mark.dependency(depends=['TestModel::test_get_compiled_model',
                                     'TestPreprocessor::test_validation_preprocess_generator', ])
    def test_model_validation_gen_compatibility(self, run_config, component_holder):
        compiled_model = component_holder['compiled_model']

        gen = component_holder['validation_data_gen']
        b = next(iter(gen))
        if len(b) == 2:
            x_b, y_b = b
            compiled_model.evaluate(x=x_b, y=y_b)
        elif len(b) == 3:
            x_b, y_b, w_b = b
            compiled_model.evaluate(x=x_b, y=y_b, sample_weight=w_b)
        else:
            raise ValueError(color_str(
                f"Generator does not return a proper number of outputs", "red",
                mode=["bold", "underline"]))
