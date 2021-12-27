Run a training pipeline
=======================

Install `abstractions`:


```shell
pip install abstractions-aimedic
```

Download mnist and re-structure the data:

```shell
datasets/
    mnist/
        train/
        validation/
        test/
```



```python

import tensorflow as tf
import tensorflow.keras as tfk

import skimage.io
import pathlib
import pandas as pd


(train_images, train_labels), (test_images, test_labels) = tfk.datasets.mnist.load_data()

train_images = train_images[:1000]
test_images = test_images[:1000]

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

def write_images(images, labels_list, img_dir):
    labels = {}
    for i, (img, l) in enumerate(zip(images, labels_list)):
        img_name = f'image_{i}.jpg'
        skimage.io.imsave(img_dir.joinpath(img_name), img)
        labels[img_name] = l

    names = list(labels.keys())
    labels = list(labels.values())

    pd.DataFrame({'image name': names, 'label': labels}).to_csv(img_dir.joinpath('labels.csv'))

data_dir = pathlib.Path('datasets').joinpath('mnist')
data_dir.mkdir(parents=True, exist_ok=True)

train_img_dir = data_dir.joinpath('train')
train_img_dir.mkdir(exist_ok=True)
write_images(train_images, train_labels, train_img_dir)

val_img_dir = data_dir.joinpath('validation')
val_img_dir.mkdir(exist_ok=True)
write_images(test_images[:500], test_labels[:500], val_img_dir)

test_img_dir = data_dir.joinpath('test')
test_img_dir.mkdir(exist_ok=True)
write_images(test_images[500:], test_labels[500:], test_img_dir)

```

clone the repository. Note that you have to put your config file in `repo_root/runs/run_name`.


```shell
git clone https://github.com/iamsoroush/mnist-test.git
```

    Cloning into 'mnist-test'...
    remote: Enumerating objects: 82, done.[K
    remote: Counting objects: 100% (82/82), done.[K
    remote: Compressing objects: 100% (41/41), done.[K
    remote: Total 82 (delta 29), reused 72 (delta 22), pack-reused 0[K
    Unpacking objects: 100% (82/82), done.



```python
run_name = 'test-1'

repo_root = pathlib.Path('mnist-test')
run_dir = repo_root.joinpath('runs').joinpath(run_name)
config_path = run_dir.joinpath('config.yaml')
```

Load config file and modify `epochs` and `dataset_dir`:


```python
from abstractions.utils import load_config_file


config = load_config_file(config_path)
config.dataset_dir = str(data_dir)
config.epochs = 10
```

Initialize `DataLoader`, `Augmentor`, `Preprocessor` and `ModelBuilder`:


```python
import sys
import os
sys.path.append(os.path.join('mnist-test', config.src_code_path))

from dataset import DataLoaderTF
from models import ModelBuilder
from preprocessing import PreprocessorTF


data_loader = DataLoaderTF(config, data_dir)
preprocessor = PreprocessorTF(config)
model_builder = ModelBuilder(config)
```

Prepare data generators:


```python
train_data_gen, train_n = data_loader.create_training_generator()
validation_data_gen, validation_n = data_loader.create_validation_generator()

train_data_gen, n_iter_train = preprocessor.add_preprocess(train_data_gen, train_n)
validation_data_gen, n_iter_val = preprocessor.add_preprocess(validation_data_gen, validation_n)
```

Start training:


```python
from abstractions.trainer import Trainer
from abstractions.utils import setup_mlflow

MLFLOW_TRACKING_URI = 'mlruns'

trainer = Trainer(config=config, run_dir=run_dir)
mlflow_active_run = setup_mlflow(mlflow_tracking_uri=MLFLOW_TRACKING_URI,
                                 mlflow_experiment_name=config.project_name,
                                 base_dir=run_dir)

trainer.train(model_builder=model_builder,
                active_run=mlflow_active_run,
                train_data_gen=train_data_gen,
                n_iter_train=n_iter_train,
                val_data_gen=validation_data_gen,
                n_iter_val=n_iter_val)
```

    2021/11/21 12:04:10 WARNING mlflow.utils.autologging_utils: You are using an unsupported version of tensorflow. If you encounter errors during autologging, try upgrading / downgrading tensorflow to a supported version, or try upgrading MLflow.
    2021/11/21 12:04:10 WARNING mlflow.utils.autologging_utils: MLflow autologging encountered a warning: "/usr/local/lib/python3.7/dist-packages/tensorflow/python/util/dispatch.py:1096: UserWarning: "`sparse_categorical_crossentropy` received `from_logits=True`, but the `output` argument was produced by a sigmoid or softmax activation and thus does not represent logits. Was this intended?""


    Epoch 1/10
    122/125 [============================>.] - ETA: 0s - loss: 0.8517 - sparse_categorical_accuracy: 0.7572INFO:tensorflow:Assets written to: mnist-test/runs/test-1/checkpoints/sm-0001-0.54541/assets
    125/125 [==============================] - 5s 15ms/step - loss: 0.8429 - sparse_categorical_accuracy: 0.7610 - val_loss: 0.5454 - val_sparse_categorical_accuracy: 0.8320
    Epoch 2/10
    119/125 [===========================>..] - ETA: 0s - loss: 0.3299 - sparse_categorical_accuracy: 0.9055INFO:tensorflow:Assets written to: mnist-test/runs/test-1/checkpoints/sm-0002-0.49706/assets
    125/125 [==============================] - 2s 13ms/step - loss: 0.3274 - sparse_categorical_accuracy: 0.9050 - val_loss: 0.4971 - val_sparse_categorical_accuracy: 0.8400
    Epoch 3/10
    120/125 [===========================>..] - ETA: 0s - loss: 0.2002 - sparse_categorical_accuracy: 0.9417INFO:tensorflow:Assets written to: mnist-test/runs/test-1/checkpoints/sm-0003-0.41642/assets
    125/125 [==============================] - 2s 16ms/step - loss: 0.1984 - sparse_categorical_accuracy: 0.9420 - val_loss: 0.4164 - val_sparse_categorical_accuracy: 0.8640
    Epoch 4/10
    116/125 [==========================>...] - ETA: 0s - loss: 0.1254 - sparse_categorical_accuracy: 0.9720INFO:tensorflow:Assets written to: mnist-test/runs/test-1/checkpoints/sm-0004-0.37957/assets
    125/125 [==============================] - 2s 13ms/step - loss: 0.1212 - sparse_categorical_accuracy: 0.9720 - val_loss: 0.3796 - val_sparse_categorical_accuracy: 0.8740
    Epoch 5/10
    120/125 [===========================>..] - ETA: 0s - loss: 0.0949 - sparse_categorical_accuracy: 0.9740INFO:tensorflow:Assets written to: mnist-test/runs/test-1/checkpoints/sm-0005-0.36565/assets
    125/125 [==============================] - 2s 16ms/step - loss: 0.0954 - sparse_categorical_accuracy: 0.9740 - val_loss: 0.3657 - val_sparse_categorical_accuracy: 0.8840
    Epoch 6/10
    120/125 [===========================>..] - ETA: 0s - loss: 0.0448 - sparse_categorical_accuracy: 0.9958INFO:tensorflow:Assets written to: mnist-test/runs/test-1/checkpoints/sm-0006-0.34909/assets
    125/125 [==============================] - 2s 16ms/step - loss: 0.0456 - sparse_categorical_accuracy: 0.9940 - val_loss: 0.3491 - val_sparse_categorical_accuracy: 0.8920
    Epoch 7/10
    117/125 [===========================>..] - ETA: 0s - loss: 0.0339 - sparse_categorical_accuracy: 0.9979INFO:tensorflow:Assets written to: mnist-test/runs/test-1/checkpoints/sm-0007-0.35331/assets
    125/125 [==============================] - 2s 13ms/step - loss: 0.0331 - sparse_categorical_accuracy: 0.9980 - val_loss: 0.3533 - val_sparse_categorical_accuracy: 0.8820
    Epoch 8/10
    119/125 [===========================>..] - ETA: 0s - loss: 0.0172 - sparse_categorical_accuracy: 0.9989INFO:tensorflow:Assets written to: mnist-test/runs/test-1/checkpoints/sm-0008-0.34437/assets
    125/125 [==============================] - 2s 16ms/step - loss: 0.0174 - sparse_categorical_accuracy: 0.9990 - val_loss: 0.3444 - val_sparse_categorical_accuracy: 0.8980
    Epoch 9/10
    113/125 [==========================>...] - ETA: 0s - loss: 0.0125 - sparse_categorical_accuracy: 1.0000INFO:tensorflow:Assets written to: mnist-test/runs/test-1/checkpoints/sm-0009-0.35300/assets
    125/125 [==============================] - 2s 13ms/step - loss: 0.0125 - sparse_categorical_accuracy: 1.0000 - val_loss: 0.3530 - val_sparse_categorical_accuracy: 0.8960
    Epoch 10/10
    120/125 [===========================>..] - ETA: 0s - loss: 0.0104 - sparse_categorical_accuracy: 1.0000INFO:tensorflow:Assets written to: mnist-test/runs/test-1/checkpoints/sm-0010-0.42965/assets
    125/125 [==============================] - 2s 14ms/step - loss: 0.0104 - sparse_categorical_accuracy: 1.0000 - val_loss: 0.4297 - val_sparse_categorical_accuracy: 0.8720


    2021/11/21 12:04:34 WARNING mlflow.utils.autologging_utils: Encountered unexpected error during tensorflow autologging: Changing param values is not allowed. Param with key='epochs' was already logged with value='50' for run ID='84655846ec2342c5a090193f808dfe34'. Attempted logging new value '10'.


Export:

```python
exported_dir = trainer._export()
```

Load the exported model and trigger evaluation process:

```python
import mlflow
from evaluation import Evaluator

exported_model = tfk.models.load_model(trainer.exported_saved_model_path)

mlflow.end_run()
eval_active_run = setup_mlflow(mlflow_tracking_uri=MLFLOW_TRACKING_URI,
                               mlflow_experiment_name=config.project_name,
                               base_dir=run_dir,
                               evaluation=True)

evaluator = Evaluator(config)
eval_report = evaluator._evaluate(data_loader=data_loader,
                                  preprocessor=preprocessor,
                                  exported_model=exported_model,
                                  active_run=eval_active_run)
eval_report.describe()


```

    100%|██████████| 500/500 [00:40<00:00, 12.21it/s]





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sparse categorical ce</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5.000000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.503735e-01</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.205305e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>9.536739e-07</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.273447e-03</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.085507e-02</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.648689e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.069383e+01</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Evaluate on validation data

val_index = data_loader.get_validation_index()
eval_report_validation = evaluator.validation_evaluate(data_loader=data_loader,
                                                        preprocessor=preprocessor,
                                                        exported_model=exported_model,
                                                        active_run=eval_active_run,
                                                        index=val_index)
```

Use `tensorboard` for visualizing training logs:


```python
tb_log_dir = trainer.tensorboard_log_dir

%load_ext tensorboard
%tensorboard --logdir {tb_log_dir}
```
