import typing

import numpy as np

from aimedeval.eval_base import SegEvalFunc


class IoUScore(SegEvalFunc):

    def __init__(self, threshold, epsilon):
        super().__init__()
        self.threshold = threshold
        self.epsilon = epsilon

    @property
    def expects_multi_channel_seg_maps(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return 'IoU Coefficient'

    @property
    def description(self) -> str:
        desc = """a function to calculate Intersection over Union score for two given masks"""
        return desc

    def get_func(self) -> typing.Callable:
        def iou_score(y_true: np.ndarray, y_pred: np.ndarray):
            """
            a function to calculate Intersection over Union score for two given masks

            Args:

                y_true: b x H x W( x Z...) x c One hot encoding of ground truth label from the dataset
                y_pred: b x H x W( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)

            Returns: Calculated iou score between y_true and y_pred, Scalar between [0, 1]
            """

            _ = self._check_inputs(y_true, y_pred)

            y_pred = (y_pred > self.threshold).astype(np.float32)
            intersection = np.logical_and(y_true, y_pred)
            union = np.logical_or(y_true, y_pred)
            iou_coef = (np.sum(intersection) + self.epsilon) / (np.sum(union) + self.epsilon)
            return iou_coef

        func = iou_score
        func.__name__ = f'iou-th{self.threshold}-ep{self.epsilon}'
        return func

    @staticmethod
    def _check_inputs(y_true, y_pred):
        assert isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray), \
            'arrays must be of type numpy.ndarray'
        assert np.issubdtype(y_true.dtype, np.number) and np.issubdtype(y_pred.dtype, np.number), \
            'the arrays data type must be numeric'
        assert y_true.shape == y_pred.shape, \
            'arrays must have equal shapes'

        if y_true.ndim == 3:
            # 2D mask
            axis = [0, 1]
        elif y_true.ndim == 4:
            # 3D mask
            axis = [0, 1, 2]
        else:
            raise Exception('arrays must either be 2D or 3D')

        return axis

