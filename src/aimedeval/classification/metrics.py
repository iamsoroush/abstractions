import typing
import tensorflow.keras as tfk
from aimedeval.eval_base import ClsEvalFunc


class CrossEntropy(ClsEvalFunc):

    def __init__(self, from_logits=False):
        super().__init__()
        self.from_logits = from_logits

    @property
    def name(self) -> str:
        return 'Cross Entropy'

    @property
    def description(self) -> str:
        desc = """a function to calculate cross entropy!"""
        return desc

    def get_func(self, ) -> typing.Callable:
        def func(y_true, y_pred):
            scce = tfk.losses.SparseCategoricalCrossentropy(from_logits=self.from_logits)
            return scce(y_true, y_pred).numpy()

        return func
