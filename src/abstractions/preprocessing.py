from abc import abstractmethod

from .base_class import BaseClass


class PreprocessorBase(BaseClass):
    """Preprocessing class.

    Responsible for re-scaling, normalization, cropping, resizing, etc.
    The output of ``add_preprocess`` method will be a genertor that generates a batch of data ``(x_batch, y_batch, sample_weight_batch/data_id_batch)``:

    - ``x_batch`` -> ``(config.batch_size, config.input_h, config.input_w, n_channels)``
    - ``y_batch`` (segmentation) -> ``(config.batch_size, config.input_h, config.input_w, n_classes)``
    - ``y_batch`` (classification) -> ``(config.batch_size, n_classes)`` or ``(config.batch_size,)``
    - ``sample_weight_batch`` (segmentation) -> ``(config.batch_size, config.input_h, config.input_w)``
    - ``sample_weight_batch`` (segmentation/classification) -> ``(config.batch_size,)``
    - ``data_id_batch`` (just for ``test_data_generator``) -> ``(config.batch_size,)``

    Notes:

        - Make sure that your ``Preprocessor`` is compatible with ``Augmentation``, ``DataLoader`` and ``ModelBuilder``.
        - Consider implementing ``image_preprocess`` and ``label_preprocess`` in a way that is compatible with serving using ``SavedModel``.
        - Note that your preprocessing has to be the same in training and inference.
        - Preprocessing will be plugged on top of ``DataLoader``'s generator, after plugging ``Augmentation``.

    Example:
        >>> preprocessor = Preprocessor(config)
        >>> data_gen = preprocessor.add_image_preprocess(data_gen)
        >>> data_gen = preprocessor.add_label_preprocess(data_gen)
        >>> data_gen, n_iter_per_epoch = preprocessor.batchify(data_gen, n_data_points)

    this is the how ``add_preprocess`` method works:

        ``add_image_preprocess`` -> ``add_label_preprocess`` -> ``batchify``

    """

    def image_preprocess(self, image):
        """Image preprocessing logic.

        Args:
            image: input image

        Returns:
            preprocessed image -> ``tf.tensor`` or ``numpy.ndarray`` of ``shape(input_height, input_width, n_channels)``
        """
        pass

    def label_preprocess(self, label):
        """Label preprocessing logic.

        Args:
            label: input label

        Returns:
            preprocessed label -> ``tf.tensor`` or ``numpy.ndarray`` of shape

                - segmentation -> ``(input_height, input_width, n_classes)``
                - classification -> ``(n_classes,)`` or scalar ``class_id``
        """
        pass

    def weight_preprocess(self, weight):
        """Weight preprocessing logic. This won't be used for ``test-data-generator``.

        Args:
            weight: input label

        Returns:
            preprocessed weight -> ``tf.tensor`` or ``numpy.ndarray`` of shape

                - segmentation -> ``(input_height, input_width, n_classes)``
                - classification -> ``(n_classes,)`` or scalar ``class_id``
        """

    @abstractmethod
    def add_image_preprocess(self, generator):
        """Plugs input-image-preprocessing on top of the given ``generator``/``tf.Dataset``.

        Notes:
            - you have to do preprocessing on image (``x``), keep label-preprocessing (if any) for ``self.add_label_preprocess``.
            - consider using the same logic that you define in ``self.image_preprocess`` method. Consider using ``map`` method for ``tf.data.Dataset``, or {``map``function/``for`` loop} for vanilla ``Python generator``.

        Args:
            generator: a ``Python generator``/``tf.data.Dataset`` which yields a single data-point ``(x, y, sample_weight)`` or ``(x, y, data_id)`` if it is ``test_data_generator`` in which

                - ``x`` => input image,
                - ``y`` => label, or segmentation map for segmentation
                - ``sample_weight`` => float (classification/segmentation), or one-channel segmentation map (segmentation)

        Returns:
            A ``generator``/``tf.data.Dataset`` with preprocessed ``x`` s, check ``self.image_preprocess`` for shapes

        """

    @abstractmethod
    def add_label_preprocess(self, generator):
        """Plugs input-label-preprocessing on top of the given ``generator``/``tf.Dataset``.

        Notes:
            - you have to do preprocessing on label (``y``), keep input-image-preprocessing (if any) for ``self.add_image_preprocess``.
            - consider using the same logic that you define in ``self.label_preprocess`` method. Consider using ``map`` method for ``tf.data.Dataset``, or {``map``function/``for`` loop} for vanilla ``Python generator``.

        Args:
            generator: a ``Python generator``/``tf.data.Dataset`` which yields a single data-point ``(x, y, sample_weight)`` or ``(x, y, data_id)`` if it is ``test_data_generator`` in which

                - ``x`` => input image,
                - ``y`` => label, or segmentation map for segmentation
                - ``sample_weight`` => float (classification/segmentation), or one-channel segmentation map (segmentation)

        Returns:
            A ``generator``/``tf.data.Dataset`` with preprocessed ``y`` s, check ``self.label_preprocess`` for shapes

        """

    @abstractmethod
    def add_weight_preprocess(self, generator):
        """Plugs input-weight-preprocessing on top of the given ``generator``/``tf.Dataset``.

        Notes:
            - you have to do preprocessing on weight (``y``), keep input image and label preprocessing (if any) for ``self.add_image_preprocess`` and ``self.add_label_preprocess``.
            - consider using exactly the same logic that you define in ``self.weight_preprocess`` method. Consider using ``map`` method for ``tf.data.Dataset``, or {``map``function/``for`` loop} for vanilla ``Python generator``.

        Args:
            generator: a ``Python generator``/``tf.data.Dataset`` which yields a single data-point ``(x, y, sample_weight)`` in which

                - ``x`` => input image,
                - ``y`` => label, or segmentation map for segmentation
                - ``sample_weight`` => float (classification/segmentation), or one-channel segmentation map (segmentation)

        Returns:
            A ``generator``/``tf.data.Dataset`` with preprocessed ``w`` s, check ``self.weight_preprocess`` for shapes

        """

    @abstractmethod
    def batchify(self, generator, n_data_points):
        """Batchifies the given ``generator``/``tf.Dataset``

        This will be the latest block for data pipeline.

        Notes:
            - you have to return ``n_iter`` in a correct way, consider a last batch of different size if ``batch_size`` does not fit into ``n_data_points``.
            - you can use ``.batch(..).repeat()`` for ``tf.data.Dataset``, or a similar process for ``Python generator``.
            - you can access to ``batch_size`` through ``self.config.batch_size``

        Args:
             generator: a ``Python generator``/``tf.data.Dataset`` which yields a single data-point ``(x, y, sample_weight)`` or ``(x, y, data_id)`` if it is ``test_data_generator`` in which

                - ``x`` =>  (input_h, input_w, n_channels),
                - ``y`` => (n_classes, 1)(classification), or (input_h, input_w, h) for segmentation
                - ``sample_weight`` => float (classification/segmentation), or one-channel segmentation map (segmentation)

            n_data_points: number of data_points in this sub-set.

        Returns:
            tuple(batched_generator, n_iter):
            - batched_generator: A repeated ``generator``/``tf.data.Dataset`` which yields a batch of data for each iteration, ``(x_batch, y_batch, sample_weights)`` or ``(x_batch, y_batch, data_ids)`` for test data gen, in which

                - ``x`` => (batch_size, input_h, input_w, n_channels)
                - ``y`` => classification: (batch_size, n_classes[or 1]), segmentation: (batch_size, input_h, input_w, n_classes)
                - ``sample_weights`` => (batch_size, 1)(classification/segmentation), (batch_size, input_h, input_w, 1)(segmentation)

            - n_iter: number of iterations per epoch

        """

    def add_preprocess(self, generator, n_data_points):
        """Plugs preprocessing on top of given generator.

        Args:
            generator: see ``DataLoaderBase.create_data_generator``
            n_data_points: number of data_points in this sub-set.

        Returns:
            tuple(batched_generator, n_iter):
            - batched_generator: A repeated ``generator``/``tf.data.Dataset`` which yields a batch of data for each iteration, ``(x_batch, y_batch, sample_weights)`` or ``(x_batch, y_batch, data_ids)`` for test data gen, in which

                - ``x`` => (batch_size, input_h, input_w, n_channels)
                - ``y`` => classification: (batch_size, n_classes[or 1]), segmentation: (batch_size, input_h, input_w, n_classes)
                - ``sample_weights`` => (batch_size, 1)(classification/segmentation), (batch_size, input_h, input_w, 1)(segmentation)

            - n_iter: number of iterations per epoch

        """

        gen = self.add_image_preprocess(generator)
        gen = self.add_label_preprocess(gen)
        gen = self.add_weight_preprocess(gen)
        gen, n_iter = self.batchify(gen, n_data_points)
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
