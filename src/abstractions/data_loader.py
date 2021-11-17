from abc import abstractmethod

from .base_class import BaseClass


class DataLoaderBase(BaseClass):
    """Data-loading mechanism.

    This class will create data generators. This is actually the process of ingesting data from a data source
     into lists of ``np.Array`` or ``tf.Tensor``, without any pre-processing(numerical manipulation of data)
    """

    @abstractmethod
    def create_training_generator(self):
        """Create data generator for training sub-set.

        This will create a ``generator``/``tf.data.Dataset`` which yields two or three components:
         (image, label, sample_weight(float))

        If you don't need ``sample_weight``, set it to ``1`` for all data-points.

        Design this generator in a way that is compatible with your implementation of ``Augmentor``, ``Preprocessor``
         and ``Evaluator``.

        Returns:
            generator: a ``generator``/``tf.data.Dataset``.
            train_n: number of training data-points.

        """

    @abstractmethod
    def create_validation_generator(self):
        """Create data generator for validation sub-set.

        This will create a ``generator``/``tf.data.Dataset`` which yields two or three components:
         (image, label, sample_weight(float))

        Design this generator in a way that is compatible with your implementation of ``Augmentor``, ``Preprocessor``
         and ``Evaluator``.

        If you don't need ``sample_weight``, set it to ``1`` for all data-points.

        Make sure that the shuffling is off for validation generator.

        If you want to add augmentation to validation data generator, define your augmentation strategy in your
         ``Augmentor`` and set ``augmentation.do_validation_augmentation==True`` in config file.

        Returns:
            generator: a ``generator``/``tf.data.Dataset``.
            val_n: number of validation data-points.

        """

    @abstractmethod
    def create_test_generator(self):
        """Create data generator for test sub-set.

        This will create a ``generator``/``tf.data.Dataset`` which yields three components:
         (image, label, data_id(str)),

        Each ``data_id`` could be anything specific that can help to retrieve this data point.
        You can consider to set ``data_id=row_id`` of the test subset's dataframe, if you are have one.
        Do not repeat this dataset, i.e. raise an exception at the end.

        Returns:
            generator: a ``generator``/``tf.data.Dataset``.
            test_n: number of test data-points.

        """

    def get_validation_index(self):
        """Returns validation index for each validation data-point.

        This will be used as ``index`` of report data-frame in evaluation process.
        Make sure that the validation generator does not shuffle, and order of this list and the validation data are
         consistent.

        Returns:
            a list of str/int indexes.

        """

        return None
