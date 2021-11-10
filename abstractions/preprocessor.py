from abc import abstractmethod

from .base_class import BaseClass


class PreprocessorBase(BaseClass):
    """Preprocessing class.

    Make sure that your ``Preprocessor`` is compatible with ``Augmentation``, ``DataLoader`` and ``ModelBuilder``.
    Consider implementing ``image_preprocess``, ``label_preprocess`` and ``batch_preprocess`` in a way that is
     compatible with serving using ``SavedModel``.

    Note that your preprocessing has to be the same in training and inference.

    """

    def image_preprocess(self, image):
        pass

    def label_preprocess(self, label):
        pass

    def add_image_preprocess(self, generator):
        pass

    def add_label_peprocess(self, generator):
        pass

    @abstractmethod
    def add_batch_preprocess(self, generator):
        """Plugs preprocessing to the end of the given ``generator``/``tf.Dataset``.

        Preprocessing will be plugged to ``DataHandler``'s generator, after plugging the ``Augmentation``.

        Attributes:
            generator: a Python generator which yields a batch of ``(x, y, sample_weights(optional))``, or
                a ``tf.data.Dataset`` which generates a batch of ``(x, y, sample_weights(optional))``.

                shapes => ``x(batch_size, h, w, n_channels)``,
                          ``y(batch_size, n_classes)``, or (batch_size, h, w, n_classes) for segmentation
                          ``sample_weights(batch_size, 1)``

        Returns:
            A ``generator``/``tf.data.Dataset`` with preprocessed elements ready to feed to the model.

        """

        image_preprocessed = self.add_image_preprocess(generator)
        label_preprocessed = self.add_label_peprocess(image_preprocessed)
        return label_preprocessed

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
