from abc import abstractmethod

from .base_class import BaseClass


class ModelBuilderBase(BaseClass):

    @abstractmethod
    def get_compiled_model(self):
        """Generates the model for training, and returns the compiled model.

        Returns:
            A compiled Keras model.
        """

        pass

    @abstractmethod
    def get_callbacks(self):
        """Returns any callbacks for ``fit``.

        Returns:
            A list of tf.keras.Callback objects. Orchestrator will handle the ModelCheckpoint and Tensorboard callbacks.
            Still, you can return each of these two callbacks, and orchestrator will modify your callbacks if needed.

        """
        return list()

    @abstractmethod
    def get_class_weight(self):
        """Set this if you want to pass ``class_weight`` to ``fit``.

        Returns:
           Optional dictionary mapping class indices (integers) to a weight (float) value,
           used for weighting the loss function (during training only). This can be useful to
           tell the model to "pay more attention" to samples from an under-represented class.

        """

        return None
    #
    # @abstractmethod
    # def post_process(self, y_pred):
    #     """Define your post-processing, used for evaluation.
    #
    #     If you have ``softmax`` as your final layer, but your labels are sparse-categorical labels, you will need to
    #      post-process the output of your model before comparing it to ``y_true``. In this case you should use
    #      ``return np.argmax(y_pred)``.
    #
    #      Note: make sure that this is compatible with your evaluation functions and ground truth labels generated
    #       using ``DataLoader``'s generators.
    #
    #      Args:
    #          y_pred: a tensor generated by ``model.predict`` method. Note that the first dimension is ``batch``.
    #
    #     Returns:
    #         post-processed batch of y_pred, ready to be compared to ground-truth.
    #
    #     """

    @classmethod
    def __subclasshook__(cls, c):
        """This defines the __subclasshook__ class method.
        This special method is called by the Python interpreter to answer the question,
         Is the class C a subclass of this class?
        """

        if cls is ModelBuilderBase:
            attrs = set(dir(c))
            if set(cls.__abstractmethods__) <= attrs:
                return True
        return NotImplemented
