import pathlib
import argparse
from pydoc import locate

import tensorflow.keras as tfk

from abstractions import *
from abstractions.utils import load_config_file, check_for_config_file, setup_mlflow

MLFLOW_TRACKING_URI = 'mlruns'


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_dir',
                        type=str,
                        help='directory of the config file',
                        required=True)
    parser.add_argument('--dataset_dir',
                        type=str,
                        help='directory of the dataset',
                        required=True)

    return parser.parse_args()


class ConfigParamDoesNotExist(Exception):
    pass


def get_class_paths(config_file):

    try:
        model_class_path = config_file.model_class
    except AttributeError:
        raise ConfigParamDoesNotExist('could not find model_class in config file.')

    try:
        preprocessor_class_path = config_file.preprocessor_class
    except AttributeError:
        raise ConfigParamDoesNotExist('could not find preprocessor_class in config file.')

    try:
        data_loader_class_path = config_file.data_loader_class
    except AttributeError:
        raise ConfigParamDoesNotExist('could not find data_loader_class in config file.')

    try:
        augmentor_class_path = config_file.augmentor_class
    except AttributeError:
        raise ConfigParamDoesNotExist('could not find augmentor_class in config file.')

    try:
        evaluator_class_path = config_file.evaluator_class
    except AttributeError:
        raise ConfigParamDoesNotExist('could not find evaluator_class in config file.')

    # try:
    #     trainer_class_path = config_file.trainer_class
    # except AttributeError:
    #     raise ConfigParamDoesNotExist('could not find trainer_class in config file.')

    return model_class_path, preprocessor_class_path, data_loader_class_path, augmentor_class_path, evaluator_class_path


if __name__ == '__main__':
    args = parse_args()

    run_dir = pathlib.Path(args.run_dir)
    config_path = check_for_config_file(run_dir)
    config_file = load_config_file(config_path.absolute())

    model_class_path, preprocessor_class_path, data_loader_class_path, augmentor_class_path, evaluator_class_path = get_class_paths(config_file)

    # Dataset
    data_loader_class = locate(data_loader_class_path)
    data_loader = data_loader_class(config_file)
    assert isinstance(data_loader, DataLoaderBase)
    train_data_gen, n_iter_train = data_loader.create_training_generator()
    # train_data_gen = data_res['data_gen']
    # train_n_iter = data_res['n_iter']
    validation_data_gen, n_iter_val = data_loader.create_validation_generator()
    # validation_data_gen = data_res['data_gen']
    # validation_n_iter = data_res['n_iter']

    # Augmentor
    augmentor_class = locate(augmentor_class_path)
    augmentor = augmentor_class(config_file)
    assert isinstance(augmentor, AugmentorBase)
    if config_file.do_train_augmentation:
        train_data_gen = augmentor.add_batch_augmentation(train_data_gen)
    if config_file.do_validation_augmentation:
        validation_data_gen = augmentor.add_validation_batch_augmentation(validation_data_gen)

    # Preprocessor
    preprocessor_class = locate(preprocessor_class_path)
    preprocessor = preprocessor_class(config_file)
    assert isinstance(preprocessor, PreprocessorBase)
    train_data_gen = preprocessor.add_batch_preprocess(train_data_gen)
    validation_data_gen = preprocessor.add_batch_preprocess(validation_data_gen)

    # Model
    model_class = locate(model_class_path)
    model_builder = model_class(config_file)

    # Trainer
    mlflow_active_run = setup_mlflow(mlflow_tracking_uri=MLFLOW_TRACKING_URI,
                                     mlflow_experiment_name=config_file.project_name,
                                     base_dir=run_dir)
    trainer = Trainer(config=config_file, run_dir=run_dir)

    # Train
    print('training ...')
    trainer.train(model_builder=model_builder,
                  active_run=mlflow_active_run,
                  train_data_gen=train_data_gen,
                  n_iter_train=n_iter_train,
                  val_data_gen=validation_data_gen,
                  n_iter_val=n_iter_val)

    # Export
    exported_dir = trainer.export()
    print(f'exported to {exported_dir}.')

    # Evaluate on evaluation data
    exported_model = tfk.models.load_model(trainer.exported_saved_model_path)

    evaluator_class = locate(evaluator_class_path)
    evaluator = evaluator_class(config_file)
    assert isinstance(evaluator, EvaluatorBase)
    evaluator.evaluate(data_loader=data_loader,
                       preprocessor=preprocessor,
                       exported_model=exported_model)

    # Evaluate on validation data


