from abc import abstractmethod

from .base_class import BaseClass


class AugmentorBase(BaseClass):
    """Augmentor class.

    Augmentations should not change the input image/label's d-type and range.
    """

    @abstractmethod
    def add_batch_augmentation(self, generator):
        """Plugs augmentation to the end of the given ``generator``/``tf.Dataset``.

        Augmentation will be plugged on ``DataHandler``'s generator, just before plugging ``Preprocessor``.

        Attributes:
            generator: a Python generator which yields a batch of ``(x, y, sample_weights(optional))``, or
                a ``tf.data.Dataset`` which generates a batch of ``(x, y, sample_weights(optional))``.

                shapes => ``x`` => a list of input images,
                          ``y`` => a list of labels, or a list of segmentation maps for segmentation
                          ``sample_weights`` => a list of labels, or a list of binary segmentation maps for segmentation

        Returns:
            A ``generator``/``tf.data.Dataset`` with augmented elements.

        """

        pass

    def add_validation_batch_augmentation(self, generator):
        """If you have different augmentation strategy for validation.

        If you don't want to ``do_validation_augmentation`` or you have similar augmentation strategies for train
         and validation, just skip over-riding this method.

        Attributes:
            generator: a Python generator which yields a batch of ``(x, y, sample_weights(optional))``, or
                a ``tf.data.Dataset`` which generates a batch of ``(x, y, sample_weights(optional))``.

                shapes => ``x`` => a list of input images,
                          ``y`` => a list of labels, or a list of segmentation maps for segmentation
                          ``sample_weights`` => a list of labels, or a list of binary segmentation maps for segmentation

        Returns:
            A ``generator``/``tf.data.Dataset`` with augmented elements.

        """

        return self.add_batch_augmentation(generator)

    @classmethod
    def __subclasshook__(cls, c):
        """This defines the __subclasshook__ class method.
        This special method is called by the Python interpreter to answer the question,
         Is the class C a subclass of this class?
        """

        if cls is AugmentorBase:
            attrs = set(dir(c))
            if set(cls.__abstractmethods__) <= attrs:
                return True
        return NotImplemented
