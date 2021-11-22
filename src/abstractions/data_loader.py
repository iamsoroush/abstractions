import pathlib
from abc import abstractmethod

from .base_class import BaseClass
from .utils import ConfigStruct


class DataLoaderBase(BaseClass):
    """Data-loading mechanism.

    This class will create data generators.
    This is actually the process of ingesting data from a data source into lists of ``np.Array`` or ``tf.Tensor``,
    without any pre-processing(numerical manipulation of data).

    Notes:
        - Output of train/validation generators will be a tuple of (image, label/segmentation_map, sample_weight). If you don't need ``sample_weight``, set it to ``1`` for all data-points.
        - Output of test generator will be a tupple of (image, label/segmentation_map, data_id). Each ``data_id`` could be anything specific that can help to retrieve this data point. You can consider to set ``data_id=row_id`` of the test subset's dataframe, if you are have one.
        - You can use the third argument with weighted metrics, or for weighted custom loss functions.

    Attributes:
        data_dir: absolute directory of the dataset
        config (ConfigStruct): config file


    Examples:
        >>> data_loader = DataLoader(config, data_dir)
        >>> data_gen, data_n = data_loader.create_training_generator()
        >>> test_data_gen, test_data_n = data_loader.create_test_generator()

    """

    def __init__(self, config, data_dir: pathlib.Path):
        self.data_dir = data_dir
        super().__init__(config)

    @abstractmethod
    def create_training_generator(self):
        """Create data generator for training sub-set.

        This will create a ``generator``/``tf.data.Dataset`` which yields three components:
        (image, label, sample_weight(s))

        Notes:
            - If you don't need ``sample_weight``, set it to ``1`` for all data-points.
            - Design this generator in a way that is compatible with your implementation of ``Augmentor``, ``Preprocessor`` and ``Evaluator``.

        Returns:
            tuple(generator, train_n):
            - generator: a ``generator``/``tf.data.Dataset``.
            - train_n: number of training data-points.

        """

    @abstractmethod
    def create_validation_generator(self):
        """Create data generator for validation sub-set.

        This will create a ``generator``/``tf.data.Dataset`` which yields three components:
        (image, label, sample_weight(s))

        Notes:
            - Design this generator in a way that is compatible with your implementation of ``Augmentor``, ``Preprocessor`` and ``Evaluator``.
            - If you don't need ``sample_weight``, set it to ``1`` for all data-points.
            - Make sure that the shuffling is off for validation generator.
            - If you want to add augmentation to validation data generator, define your augmentation strategy in your ``Augmentor`` and set ``augmentation.do_validation_augmentation=True`` in config file.

        Returns:
            tuple(generator, val_n):
            - generator: a ``generator``/``tf.data.Dataset``.
            - val_n: number of validation data-points.

        """

    @abstractmethod
    def create_test_generator(self):
        """Create data generator for test sub-set.

        This will create a ``generator``/``tf.data.Dataset`` which yields three components:
        (image, label, data_id(str)),

        Notes:
            - Each ``data_id`` could be anything specific that can help to retrieve this data point.
            - You can consider to set ``data_id=row_id`` of the test subset's dataframe, if you are have one.
            - Do not repeat this dataset, i.e. raise an exception at the end.

        Returns:
            tuple(generator, test_n):
            - generator: a ``generator``/``tf.data.Dataset``.
            - test_n: number of test data-points.

        """

    def get_validation_index(self):
        """Returns validation index for each validation data-point.

        This will be used as ``index`` of report-data-frame in evaluation process.

        Notes:
            Make sure that the validation generator does not shuffle, and order of this list and the validation data are the same.

        Returns:
            list of str/int indexes.

        """

        return None
