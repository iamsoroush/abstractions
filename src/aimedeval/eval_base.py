from abc import abstractmethod
import typing


class EvalFunc:

    """Rule: function's input and output must be ``np.ndarray``"""

    @abstractmethod
    @property
    def name(self) -> str:
        pass

    @abstractmethod
    @property
    def description(self) -> str:
        pass
        # return self.__doc__

    @abstractmethod
    def get_func(self) -> typing.Callable:
        pass


class SegEvalFunc(EvalFunc):

    def expected_shape(self) -> typing.Tuple[typing.Union[str, int]]:
        """function expects y_true and y_pred to be in this shape.

         For example, if your eval-func expects a one-channel binary segmentation map: ``('height', 'width', 1)``, or
         if you expect batches, return ``('batch_size', 'height', 'width', 1)``
         """

        ret = list()
        if self.expects_batch:
            ret.append('batch')
        ret.append('height')
        ret.append('width')
        n_channels = self.expected_seg_map_channels
        if n_channels is not None:
            ret.append(n_channels)
        return tuple(ret)

    @abstractmethod
    @property
    def expected_seg_map_channels(self) -> typing.Optional[int]:
        """return ``None`` if you expect something like (batch, h, w), and return num-channels if you expect something like (batch, h, w, 1) """
        pass

    @abstractmethod
    @property
    def expects_batch(self) -> bool:
        """If this function expects batches of data, i.e. the first dimension is ``batch-size``"""

        pass

    @abstractmethod
    @property
    def expected_input_type(self) -> str:
        """if your functions assumes that the y_pred and y_true are:
            numpy array -> return ``'np'``
            tensorflow tensor -> return ``'tf'``
        """
        pass
