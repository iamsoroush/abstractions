from abc import abstractmethod

from .base_class import BaseClass


class PreprocessorBase(BaseClass):
    """Preprocessing class.

    Make sure that your ``Preprocessor`` is compatible with ``Augmentation``, ``DataLoader`` and ``ModelBuilder``.
    Consider implementing ``image_preprocess`` and ``label_preprocess`` in a way that is
     compatible with serving using ``SavedModel``.

    Note that your preprocessing has to be the same in training and inference.

    """

    @abstractmethod
    def image_preprocess(self, image):
        pass

    @abstractmethod
    def label_preprocess(self, label):
        pass

    @abstractmethod
    def add_image_preprocess(self, generator):
        """Plugs input-image-preprocessing to the end of the given ``generator``/``tf.Dataset``.

        Preprocessing will be plugged to ``DataHandler``'s generator, after plugging the ``Augmentation``.

        Note that you have to do preprocessing on image (``x``), keep label-preprocessing (if any) for
         ``self.add_label_preprocess``.

        Note that you have to use exactly the same logic that you define in ``self.image_preprocess`` method. Consider
         using ``map`` method for ``tf.data.Dataset``, or ``map`` function for vanilla ``Python generator``.

        Attributes:
            generator: a ``Python generator``/``tf.data.Dataset`` which yields a single data-point
             ``(x, y, sample_weight)`` or ``(x, y, data_id)`` if it is ``test_data_generator``

                shapes => ``x`` => input image,
                          ``y`` => label, or segmentation map for segmentation
                          ``sample_weight`` => float (classification), or binary segmentation map (segmentation)

        Returns:
            A ``generator``/``tf.data.Dataset`` with preprocessed ``x``s.

        """

    @abstractmethod
    def add_label_preprocess(self, generator):
        """Plugs input-image-preprocessing to the end of the given ``generator``/``tf.Dataset``.

        Preprocessing will be plugged to ``DataHandler``'s generator, after plugging the ``Augmentation``.

        Note that you have to do preprocessing on label (``y``), keep input-image-preprocessing (if any) for
         ``self.add_image_preprocess``.

        Note that you have to use exactly the same logic that you define in ``self.label_preprocess`` method. Consider
         using ``map`` method for ``tf.data.Dataset``, or ``map`` function for vanilla ``Python generator``.

        Attributes:
            generator: a ``Python generator``/``tf.data.Dataset`` which yields a single data-point
             ``(x, y, sample_weight)`` or ``(x, y, data_id)`` if it is ``test_data_generator``

                shapes => ``x`` => input image,
                          ``y`` => label, or segmentation map for segmentation
                          ``sample_weight`` => float (classification), or binary segmentation map (segmentation)

        Returns:
            A ``generator``/``tf.data.Dataset`` with preprocessed ``x``s.

        """

    @abstractmethod
    def batchify(self, generator, n_data_points, batch_size):
        """Batchifies the input ``generator``/``tf.Dataset``

        This will be the latest block for data pipeline.
        Note that you have to return the ``n_iter`` in a correct way, consider the last batch of different size if
         the ``batch_size`` does not fit into the ``n_data_points``.

        Args:
             generator: a ``Python generator``/``tf.data.Dataset`` which yields a single data-point
             ``(x, y, sample_weight)`` or ``(x, y, data_id)`` if it is ``test_data_generator``

                shapes => ``x`` =>  (input_h, input_w, n_channels),
                          ``y`` => (n_classes, 1)(classification), or (input_h, input_w, h) for segmentation
                          ``sample_weight`` => float (classification), or binary segmentation map (segmentation)

            n_data_points: number of data_points for this sub-set.
            batch_size: batch size

        Returns:
            batched_generator: A ``generator``/``tf.data.Dataset`` which yields a batch of data for each iteration:
                ``(x_batch, y_batch, sample_weights)`` or ``(x_batch, y_batch, data_ids)`` for test data gen.
                shapes => ``x`` => (batch_size, input_h, input_w, n_channels)
                          ``y`` => classification: (batch_size, n_classes), segmentation: (batch_size, input_h, input_w, n_classes)
                          ``sample_weights`` => (batch_size, 1)(classification), (batch_size, input_h, input_w, n_classes)(segmentation)
            n_iter: number of iterations per epoch

        """

    def add_preprocess(self, generator, n_data_points, batch_size):
        gen = self.add_image_preprocess(generator)
        gen = self.add_label_preprocess(gen)
        gen, n_iter = self.batchify(gen, n_data_points, batch_size)
        return gen, n_iter

    @classmethod
    def __subclasshook__(cls, c):
        """This defines the __subclasshook__ class method.
        This special method is called by the Python interpreter to answer the question,
         Is the class C a subclass of this class?
        """

        if cls is PreprocessorBase:
            attrs = set(dir(c))
            if set(cls.__abstractmethods__) <= attrs:
                return True
        return NotImplemented
