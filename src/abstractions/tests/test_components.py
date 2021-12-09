import pytest
from pathlib import Path
from pydoc import locate
import numpy as np

from abstractions.utils import load_config_file
from abstractions import DataLoaderBase, AugmentorBase, PreprocessorBase, ModelBuilderBase, EvaluatorBase
from types import FunctionType


@pytest.fixture(scope='module')
def component_holder():
    return {}


@pytest.fixture()
def run_config(pytestconfig):
    config_path = pytestconfig.getoption('config_path')
    if config_path is None:
        run_name = pytestconfig.getoption('run_name')
        run_dir = Path('runs').joinpath(run_name)
        config = load_config_file(list(run_dir.glob('*.yaml'))[0])
    else:
        config = load_config_file(config_path)
    return config


# @pytest.mark.component
# @pytest.mark.dependency
# class TestConfigFile:


@pytest.mark.component
@pytest.mark.dependency
class TestLocateComponents:

    def test_locate_data_loader_class(self, run_config, component_holder):
        data_loader_class = locate(run_config.data_loader_class)
        assert data_loader_class is not None
        component_holder['data_loader_class'] = data_loader_class

    def test_locate_augmentor_class(self, run_config, component_holder):
        if not hasattr(run_config, 'augmentor_class'):
            assert True
        else:
            augmentor_class = locate(run_config.augmentor_class)
            assert augmentor_class is not None
            component_holder['augmentor_class'] = augmentor_class

    def test_locate_preprocessor_class(self, run_config, component_holder):
        preprocessor_class = locate(run_config.preprocessor_class)
        assert preprocessor_class is not None
        component_holder['preprocessor_class'] = preprocessor_class

    def test_locate_model_builder_class(self, run_config, component_holder):
        model_builder_class = locate(run_config.model_builder_class)
        assert model_builder_class is not None
        component_holder['model_builder_class'] = model_builder_class

    def test_locate_evaluator_class(self, run_config, component_holder):
        evaluator_class = locate(run_config.evaluator_class)
        assert evaluator_class is not None
        component_holder['evaluator_class'] = evaluator_class


@pytest.mark.component
class TestInitializeComponents:

    @pytest.mark.dependency(depends=['TestLocateComponents::test_locate_data_loader_class'])
    def test_initialize_data_loader(self, run_config, component_holder):
        data_dir = Path(run_config.data_dir)
        data_loader = component_holder.get('data_loader_class')(run_config, data_dir)
        assert isinstance(data_loader, DataLoaderBase)
        component_holder['data_loader'] = data_loader

    @pytest.mark.dependency(depends=['TestLocateComponents::test_locate_augmentor_class'])
    def test_initialize_augmentor(self, run_config, component_holder):
        augmentor_class = component_holder.get('augmentor_class')
        if augmentor_class is not None:
            augmentor = augmentor_class(run_config)
            assert isinstance(augmentor, AugmentorBase)
            component_holder['augmentor'] = augmentor

    @pytest.mark.dependency(depends=['TestLocateComponents::test_locate_preprocessor_class'])
    def test_initialize_preprocessor(self, run_config, component_holder):
        preprocessor = component_holder.get('preprocessor_class')(run_config)
        assert isinstance(preprocessor, PreprocessorBase)
        component_holder['preprocessor'] = preprocessor

    @pytest.mark.dependency(depends=['TestLocateComponents::test_locate_model_builder_class'])
    def test_initialize_model_builder(self, run_config, component_holder):
        model_builder = component_holder.get('model_builder_class')(run_config)
        assert isinstance(model_builder, ModelBuilderBase)
        component_holder['model_builder'] = model_builder

    @pytest.mark.dependency(depends=['TestLocateComponents::test_locate_evaluator_class'])
    def test_initialize_evaluator(self, run_config, component_holder):
        evaluator = component_holder.get('evaluator_class')(run_config)
        assert isinstance(evaluator, EvaluatorBase)
        component_holder['evaluator'] = evaluator


@pytest.mark.component
class TestDataLoader:

    # ---------- Train Data ----------
    @pytest.mark.dependency(depends=['TestInitializeComponents::test_initialize_data_loader'])
    def test_create_training_generator(self, run_config, component_holder):
        data_loader = component_holder['data_loader']
        ret = data_loader.create_training_generator()
        assert len(ret) == 2

        train_data_gen, train_n = ret
        component_holder['train_data_gen'] = train_data_gen
        component_holder['train_n'] = train_n

    @pytest.mark.dependency(depends=['TestDataLoader::test_create_training_generator'])
    def test_train_n(self, run_config, component_holder):
        assert isinstance(component_holder['train_n'], int)

    @pytest.mark.dependency(depends=['TestDataLoader::test_create_training_generator'])
    def test_train_generator(self, run_config, component_holder):
        assert hasattr(component_holder['train_data_gen'], '__iter__')

    @pytest.mark.dependency(depends=['TestDataLoader::test_train_generator'])
    def test_train_gen_out(self, component_holder):
        gen = component_holder['train_data_gen']
        ret = next(iter(gen))
        assert len(ret) == 3

    # ---------- Validation Data ----------
    @pytest.mark.dependency(depends=['TestInitializeComponents::test_initialize_data_loader'])
    def test_create_validation_generator(self, run_config, component_holder):
        data_loader = component_holder['data_loader']
        ret = data_loader.create_validation_generator()
        assert len(ret) == 2

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
        assert len(ret) == 3

    # ---------- Test Data ----------
    @pytest.mark.dependency(depends=['TestInitializeComponents::test_initialize_data_loader'])
    def test_create_evaluation_generator(self, run_config, component_holder):
        data_loader = component_holder['data_loader']
        ret = data_loader.create_test_generator()
        assert len(ret) == 2

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
        assert len(ret) == 3


@pytest.mark.component
class TestAugmentor:

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
        train_data_gen = component_holder['train_data_gen']
        ret = next(iter(train_data_gen))
        assert len(ret) == 3

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
        data_gen = component_holder['validation_data_gen']
        ret = next(iter(data_gen))
        assert len(ret) == 3


@pytest.mark.component
class TestPreprocessor:

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
        assert len(ret) == 3

        x_batch, y_batch, w_batch = ret
        component_holder['train_gen_xb_shape'] = tuple(x_batch.shape)
        component_holder['train_gen_yb_shape'] = tuple(y_batch.shape)
        component_holder['train_gen_wb'] = w_batch

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
        w_batch = component_holder['train_gen_wb']
        assert len(w_batch) == int(run_config.batch_size)

    @pytest.mark.dependency(depends=['TestPreprocessor::test_train_gen_out'])
    def test_train_gen_w_is_iterable(self, run_config, component_holder):
        w_batch = component_holder['train_gen_wb']
        assert hasattr(w_batch[0],
                       '__iter__'), 'elements of the weights_batch are not iterables, add a new axis to each element.'

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
        assert len(ret) == 3

        x_batch, y_batch, w_batch = ret
        component_holder['validation_gen_xb_shape'] = tuple(x_batch.shape)
        component_holder['validation_gen_yb_shape'] = tuple(y_batch.shape)
        component_holder['validation_gen_wb'] = w_batch

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
        w_batch = component_holder['validation_gen_wb']
        assert len(w_batch) == int(run_config.batch_size)

    @pytest.mark.dependency(depends=['TestPreprocessor::test_validation_gen_out'])
    def test_validation_gen_w_is_iterable(self, run_config, component_holder):
        w_batch = component_holder['validation_gen_wb']
        assert hasattr(w_batch[0],
                       '__iter__'), 'elements of the weights_batch are not iterables, add a new axis to each element.'

    # ---------- Evaluation Preprocessing ----------
    @pytest.mark.dependency(depends=['TestDataLoader::test_evaluation_generator'])
    def test_add_evaluation_preprocessing(self, run_config, component_holder):
        preprocessor = component_holder['preprocessor']
        data_gen = component_holder['evaluation_data_gen']
        n = component_holder['evaluation_n']

        # ret = preprocessor.add_preprocess(data_gen, n)
        gen = preprocessor.add_image_preprocess(data_gen)
        gen = preprocessor.add_label_preprocess(gen)
        ret = preprocessor.batchify(gen, n)

        assert len(ret) == 2

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
        assert len(ret) == 3

        x_batch, y_batch, w_batch = ret
        component_holder['evaluation_gen_xb_shape'] = tuple(x_batch.shape)
        component_holder['evaluation_gen_yb_shape'] = tuple(y_batch.shape)
        component_holder['evaluation_gen_id'] = w_batch

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
    def test_evaluation_gen_w_batch_size(self, run_config, component_holder):
        id_batch = component_holder['evaluation_gen_id']
        assert len(id_batch) == int(run_config.batch_size)

    # @pytest.mark.dependency(depends=['TestPreprocessor::test_evaluation_gen_out'])
    # def test_evaluation_gen_w_is_iterable(self, run_config, component_holder):
    #     w_batch = component_holder['evaluation_gen_wb']
    #     assert hasattr(w_batch[0],
    #                    '__iter__'), 'elements of the weights_batch are not iterables, add a new axis to each element.'


@pytest.mark.component
class TestModelBuilder:

    @pytest.mark.dependency(depends=['TestPreprocessor::test_train_preprocess_generator'])
    def test_get_compiled_model(self, run_config, component_holder):
        model_builder = component_holder['model_builder']
        compiled_model = model_builder.get_compiled_model()
        assert True  # TODO: check for being a tensorflow.keras model

        component_holder['compiled_model'] = compiled_model

    @pytest.mark.dependency(depends=['TestModelBuilder::test_get_compiled_model',
                                     'TestPreprocessor::test_train_preprocess_generator', ])
    def test_model_train_gen_compatibility(self, run_config, component_holder):
        compiled_model = component_holder['compiled_model']
        _, input_h, input_w, n_channels = compiled_model.input_shape

        train_gen = component_holder['train_data_gen']
        x_b, y_b, w_b = next(iter(train_gen))

        compiled_model.evaluate(x=x_b, y=y_b, sample_weight=w_b)
        assert True

        # assert input_h == int(run_config.input_height)
        # assert input_w == int(run_config.input_width)
        # assert n_channels == int(component_holder['input_channels'])

    @pytest.mark.dependency(depends=['TestModelBuilder::test_get_compiled_model',
                                     'TestPreprocessor::test_validation_preprocess_generator', ])
    def test_model_validation_gen_compatibility(self, run_config, component_holder):
        compiled_model = component_holder['compiled_model']
        _, input_h, input_w, n_channels = compiled_model.input_shape

        gen = component_holder['validation_data_gen']
        x_b, y_b, w_b = next(iter(gen))

        compiled_model.evaluate(x=x_b, y=y_b, sample_weight=w_b)
        assert True

        # assert input_h == int(run_config.input_height)
        # assert input_w == int(run_config.input_width)
        # assert n_channels == int(component_holder['input_channels'])


@pytest.mark.component
class TestEvaluation:

    @pytest.mark.dependency(depends=['TestInitializeComponents::test_initialize_evaluator',
                                     'TestPreprocessor::test_evaluation_gen_out'])
    def test_eval_funcs_on_evaluation_gen(self, component_holder):
        evaluator = component_holder['evaluator']
        eval_funcs = evaluator.get_eval_funcs()

        compiled_model = component_holder['compiled_model']
        data_gen = component_holder['evaluation_data_gen']
        x_b, y_b, _ = next(iter(data_gen))
        x_sample = x_b[0]
        y_sample = y_b[0]
        y_pred = compiled_model.predict(np.expand_dims(x_sample, axis=0))[0]

        failed_funcs = list()
        for f_name, f in eval_funcs.items():
            if not (isinstance(f_name, str) or isinstance(f, FunctionType)):
                failed_funcs.append(f_name)
            else:
                try:
                    f(y_sample, y_pred)
                except Exception as e:
                    failed_funcs.append(f_name)

        if any(failed_funcs):
            pytest.fail(f'failed {len(failed_funcs)}/{len(eval_funcs)}. failed funcs are {failed_funcs}')

    def test_eval_funcs_on_validation_gen(self, component_holder):
        evaluator = component_holder['evaluator']
        eval_funcs = evaluator.get_eval_funcs()

        compiled_model = component_holder['compiled_model']
        data_gen = component_holder['validation_data_gen']
        x_b, y_b, _ = next(iter(data_gen))
        x_sample = x_b[0]
        y_sample = y_b[0]
        y_pred = compiled_model.predict(np.expand_dims(x_sample, axis=0))[0]

        failed_funcs = list()
        for f_name, f in eval_funcs.items():
            if not (isinstance(f_name, str) or isinstance(f, FunctionType)):
                failed_funcs.append(f_name)
            else:
                try:
                    f(y_sample, y_pred)
                except Exception as e:
                    failed_funcs.append(f_name)

        if any(failed_funcs):
            pytest.fail(f'failed {len(failed_funcs)}/{len(eval_funcs)}. failed funcs are {failed_funcs}')

    # @pytest.mark.dependency(depends=['TestEvaluation::test_get_eval_funcs'])
    # def test_eval_funcs(self, component_holder):
    #     compiled_model = component_holder['compiled_model']
    #     eval_funcs = component_holder['eval_funcs']
    #     data_gen = component_holder['train_data_gen']
    #     x_b, y_b, _ = next(iter(data_gen))
    #     x_sample = x_b[0]
    #     y_sample = y_b[0]
    #     y_pred = compiled_model.predict(np.expand_dims(x_sample, axis=0))[0]
    #
    #     for f_name, f in eval_funcs.items():
    #         try:
    #             f(y_sample, y_pred)
    #         except Exception as e:
    #             pytest.fail(f'eval func {f_name} failed with exception {e}')


@pytest.mark.component
class TestTraining:

    @pytest.mark.dependency(depends=['TestEvaluation::test_eval_funcs_on_evaluation_gen',
                                     'TestEvaluation::test_eval_funcs_on_validation_gen'])
    def test_model_training(self, run_config, component_holder):
        """Simple model fit, with 3 batches per epoch, for 3 epochs."""

        model = component_holder['compiled_model']
        train_gen = component_holder['train_data_gen']
        val_gen = component_holder['validation_data_gen']

        x_tr, y_tr, w_tr = next(iter(train_gen))
        x_val, y_val, w_val = next(iter(val_gen))

        initial_tr_loss = model.evaluate(x_tr, y_tr, sample_weight=w_tr, return_dict=True)['loss']
        initial_val_loss = model.evaluate(x_val, y_val, sample_weight=w_val, return_dict=True)['loss']
        # train_iter = 3  # component_holder['n_iter_train']
        # val_iter = 3  # component_holder['n_iter_validation']

        epochs = 3

        history = model.fit(x_tr,
                            y_tr,
                            epochs=epochs,
                            sample_weight=w_tr,
                            validation_data=(x_val, y_val, w_val))
        assert True

        component_holder['training_history'] = history.history
        component_holder['initial_training_loss'] = initial_tr_loss
        component_holder['initial_validation_loss'] = initial_val_loss

    @pytest.mark.dependency(depends=['TestTraining::test_model_training'])
    def test_model_loss_is_decreasing(self, component_holder):
        model_loss = component_holder['training_history']['loss']
        assert model_loss[-1] < model_loss[0]

    @pytest.mark.dependency(depends=['TestTraining::test_model_training'])
    def test_model_val_loss_is_decreasing(self, component_holder):
        model_loss = component_holder['training_history']['val_loss']
        assert model_loss[-1] < model_loss[0]
