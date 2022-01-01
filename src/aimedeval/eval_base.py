from abc import abstractmethod
import typing


class EvalFunc:

    """Rule: function's input and output must be ``np.ndarray``"""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass
        # return self.__doc__

    @abstractmethod
    def get_func(self) -> typing.Callable:
        pass


class SegEvalFunc(EvalFunc):

    # input and output type: numpy, no batch

    def expected_shape(self) -> typing.Tuple[str]:
        """function expects y_true and y_pred to be in this shape.

         For example, if your eval-func expects a one-channel binary segmentation map: ``('height', 'width', 1)``
         """

        ret = list()

        ret.append('height')
        ret.append('width')
        if self.expects_multi_channel_seg_maps:
            ret.append('n_classes')
        return tuple(ret)

    @property
    @abstractmethod
    def expects_multi_channel_seg_maps(self) -> bool:
        """return ``True`` if your metric expects something like (h, w, n), and ``False`` if you expect binary maps which
         have 2 dimensions, i.e (h, w)"""
        pass

    # @abstractmethod
    # @property
    # def expects_batch(self) -> bool:
    #     """If this function expects batches of data, i.e. the first dimension is ``batch-size``"""
    #
    #     pass
    #
    # @abstractmethod
    # @property
    # def expected_input_type(self) -> str:
    #     """if your functions assumes that the y_pred and y_true are:
    #         numpy array -> return ``'np'``
    #         tensorflow tensor -> return ``'tf'``
    #     """
    #     pass
