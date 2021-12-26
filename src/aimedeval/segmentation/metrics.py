import typing

import numpy as np
from sklearn.metrics import confusion_matrix

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

                y_true: H x W( x Z...) x c One hot encoding of ground truth label from the dataset
                y_pred: H x W( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)

            Returns: Calculated iou score between y_true and y_pred, Scalar between [0, 1]
            """

            _ = _check_inputs(y_true, y_pred)

            y_pred = (y_pred > self.threshold).astype(np.float32)
            intersection = np.logical_and(y_true, y_pred)
            union = np.logical_or(y_true, y_pred)
            iou_coef = (np.sum(intersection) + self.epsilon) / (np.sum(union) + self.epsilon)
            return iou_coef

        func = iou_score
        func.__name__ = f'iou-th{self.threshold}-ep{self.epsilon}'
        return func


class SoftIoUScore(SegEvalFunc):

    @property
    def expects_multi_channel_seg_maps(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "Soft-IoU"

    @property
    def description(self) -> str:
        return "Calculates the intersection over union score without thresholding the y_pred."\
               "This metric expects one hot encoding of ground truth label and soft-maxed output of" \
               " a segmentation model, and returns a value in [0, 1] range."

    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon

    def get_func(self) -> typing.Callable:

        def soft_iou_score(y_true, y_pred):
            """
            Calculates the intersection over union score without one-hot encoding of y_pred

            Args:
                y_true (): H x W( x Z...) x c One hot encoding of ground truth label from the dataset
                y_pred (): H x W( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)

            Returns: Calculated iou score between y_true and y_pred, Scalar between [0, 1]
            """

            _ = _check_inputs(y_true, y_pred)

            y_true = np.array(y_true, dtype=np.float32)
            y_pred = np.array(y_pred, dtype=np.float32)

            intersection = np.sum(np.abs(y_true * y_pred))
            union = np.sum(y_true) + np.sum(y_pred) - intersection
            iou = np.mean((intersection + self.epsilon) / (union + self.epsilon))

            return iou

        func = soft_iou_score
        func.__name__ = f'soft-iou-ep{self.epsilon}'
        return func


class DiceScore(SegEvalFunc):

    @property
    def expects_multi_channel_seg_maps(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "Dice Coefficient"

    @property
    def description(self) -> str:
        return "Calculates the Dice score which is an iou-like metric robust to class imbalanced samples."\
               "This metric expects one-hot-encoding of ground truth label and soft-maxed output of" \
               " a segmentation model, and returns a value in [0, 1] range."

    def __init__(self, threshold=0.5, epsilon=1e-6):
        super().__init__()
        self.threshold = threshold
        self.epsilon = epsilon

    def get_func(self) -> typing.Callable:

        def dice_score(y_true, y_pred):
            """

            calculates the dice score between y_true and y_pred
            Args:
                y_true (): H x W( x Z...) x c One hot encoding of ground truth label from the dataset
                y_pred (): H x W( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)

            Returns: dice score between y_true and y_pred

            """

            _ = _check_inputs(y_true, y_pred)

            y_pred = (y_pred > self.threshold).astype(np.float32)
            intersection = np.logical_and(y_true, y_pred)
            intersection2 = 2 * np.sum(intersection)
            dice_coef = (intersection2 + self.epsilon) / (np.sum(y_pred + y_true) + self.epsilon)

            return dice_coef

        func = dice_score
        func.__name__ = f'dice-th{self.threshold}-ep{self.epsilon}'
        return func


class SoftDice(SegEvalFunc):

    @property
    def expects_multi_channel_seg_maps(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "Soft Dice Coefficient"

    @property
    def description(self) -> str:
        return "Calculates the soft-Dice-score which is an iou-like metric robust to class imbalanced samples, " \
               "calculated on y_pred without thresholding." \
               "This metric expects one-hot-encoding of ground truth label and soft-maxed output of" \
               " a segmentation model, and returns a value in [0, 1] range."

    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def get_func(self) -> typing.Callable:

        def soft_dice_score(y_true, y_pred):
            """

            Args:
                y_true (): b x H x W( x Z...) x c One hot encoding of ground truth label from the dataset
                y_pred (): b x H x W( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)
                epsilon (): a small scalar value used to avoid zero-division errors

            Returns: soft dice score between y_true and y_pred

            """

            _ = _check_inputs(y_true, y_pred)

            y_true = np.array(y_true, dtype=np.float32)
            y_pred = np.array(y_pred, dtype=np.float32)

            numerator = 2. * np.sum(y_pred * y_true)
            denominator = np.sum(np.square(y_pred) + np.square(y_true))
            soft_dice = np.mean((numerator + self.epsilon) / (denominator + self.epsilon))

            return soft_dice

        func = soft_dice_score
        func.__name__ = f'dice-ep{self.epsilon}'
        return func


class TPR(SegEvalFunc):

    @property
    def expects_multi_channel_seg_maps(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "True Positive Rate (Recall)"

    @property
    def description(self) -> str:
        return "Calculates the TPR, i.e. the TP/(TP+FN). " \
               "This metric expects one-hot-encoding of ground truth label and soft-maxed output of" \
               " a segmentation model, and returns a value in [0, 1] range."

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def get_func(self) -> typing.Callable:

        def true_positive_rate(y_true, y_pred):
            """

            Args:
                y_true ():
                y_pred ():

            Returns:
                 the percentage of the true positives in range [0, 1]

            """
            _ = _check_inputs(y_true, y_pred)
            tp, tn, fp, fn = get_conf_mat_elements(y_true, y_pred, self.threshold)
            return float((tp / (tp + fn)) * 100)

        f = true_positive_rate
        f.__name__ += f'_th{self.threshold}'

        return f


class TNR(SegEvalFunc):

    @property
    def expects_multi_channel_seg_maps(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "True Negative Rate"

    @property
    def description(self) -> str:
        return "Calculates the TNR, i.e. the TN/(FP+TN). " \
               "This metric expects one-hot-encoding of ground truth label and soft-maxed output of" \
               " a segmentation model, and returns a value in [0, 1] range."

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def get_func(self) -> typing.Callable:

        def true_negative_rate(y_true, y_pred):
            """

            Args:
                y_true ():
                y_pred ():

            Returns:
                the percentage of the true negatives in range [0, 1]

            """
            _ = _check_inputs(y_true, y_pred)
            tp, tn, fp, fn = get_conf_mat_elements(y_true, y_pred, self.threshold)
            return float((tn / (fp + tn)) * 100)

        f = true_negative_rate
        f.__name__ += f'_th{self.threshold}'

        return f


class Precision(SegEvalFunc):

    @property
    def expects_multi_channel_seg_maps(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "Precision"

    @property
    def description(self) -> str:
        return "Calculates the Precision, i.e. the TP/(FP+TP). " \
               "This metric expects one-hot-encoding of ground truth label and soft-maxed output of" \
               " a segmentation model, and returns a value in [0, 1] range."

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def get_func(self) -> typing.Callable:

        def precision(y_true, y_pred):
            """

            Args:
                y_true ():
                y_pred ():

            Returns:
                the percentage of the true negatives in range [0, 1]

            """
            _ = _check_inputs(y_true, y_pred)
            tp, tn, fp, fn = get_conf_mat_elements(y_true, y_pred, self.threshold)
            return float((tp / (fp + tp)) * 100)

        f = precision
        f.__name__ += f'_th{self.threshold}'

        return f


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


def get_conf_mat_elements(y_true, y_pred, threshold=0.5):
    """
    Returns true positives count
    Args:
        y_true (): array of shape(input_h, input_w, 1).dtype(tf.float32) and {0, 1}
        y_pred (): array of shape(input_h, input_w, 1).dtype(tf.float32) and [0, 1]
        threshold (): threshold on ``y_pred``

    Returns: tp: true positives count
             tn: true negatives count
             fp: false positives count
             fn: false negatives count

    """

    _ = _check_inputs(y_true, y_pred)

    y_pred_thresholded = (y_pred > threshold).astype(np.float32)
    y_true_thresholded = (y_true > threshold).astype(np.float32)

    conf_mat = confusion_matrix(np.reshape(y_true_thresholded, -1), np.reshape(y_pred_thresholded, -1))
    tn, fp, fn, tp = np.reshape(conf_mat, -1)
    return tp, tn, fp, fn

