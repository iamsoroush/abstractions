from abc import abstractmethod

from .base_class import BaseClass


class AugmentorBase(BaseClass):
    """Augmentor class.

    Notes:
        - Augmentations should not change the input image/label's d-type, size and range, any modifications that need to be the same in inference adn training must be implemented in ``Preprocessor``
        - Make sure that label/image augmentation are compatible.

    Examples:
        >>> augmentor = Augmentor(config)
        >>> data_gen = augmentor.add_augmentation(data_gen)

    """

    @abstractmethod
    def add_augmentation(self, generator):
        """Plugs augmentation to the end of the given ``generator``/``tf.Dataset``.

        See ``DataLoader`` for more info about the input generator.

        Args:
            generator: a ``Python generator`` or ``tf.data.Dataset`` which yields a single data-point of
             ``(x, y, sample_weight(s))`` or ``(x, y, data_id)`` if it is ``test_data_generator``.

                - ``x``: input image,
                - ``y``: label, or segmentation map (segmentation)
                - ``sample_weight``: float (classification/segmentation), or binary segmentation map (segmentation)

        Returns:
            A ``generator``/``tf.data.Dataset`` with augmented elements.

        """

    @abstractmethod
    def add_validation_augmentation(self, generator):
        """If you have different augmentation strategy for validation.

        If you don't want to ``do_validation_augmentation`` or you have similar augmentation strategies for train
        and validation, just skip over-riding this method.

        Args:
            generator: a ``Python generator`` or ``tf.data.Dataset`` which yields a single data-point of
             ``(x, y, sample_weight(s))`` or ``(x, y, data_id)`` if it is ``test_data_generator``.

                - ``x``: input image,
                - ``y``: label, or segmentation map (segmentation)
                - ``sample_weight``: float (classification/segmentation), or binary segmentation map (segmentation)

        Returns:
            A ``generator``/``tf.data.Dataset`` with augmented elements.

        """

        return self.add_augmentation(generator)

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
