from abc import abstractmethod

from .base_class import BaseClass
from .data_loader import DataLoaderBase
from .preprocessor import PreprocessorBase


class EvaluatorBase(BaseClass):

    def evaluate(self, data_loader: DataLoaderBase, preprocessor: PreprocessorBase, compiled_model):
        """Evaluates the model using ``eval_functions`` defined in ``get_eval_functions``."""

        test_data_gen, n_iter_test = data_loader.create_test_generator()

        test_data_gen = preprocessor.add_batch_preprocess(test_data_gen)

    @classmethod
    def __subclasshook__(cls, c):
        """This defines the __subclasshook__ class method.
        This special method is called by the Python interpreter to answer the question,
         Is the class C a subclass of this class?
        """

        if cls is EvaluatorBase:
            attrs = set(dir(c))
            if set(cls.__abstractmethods__) <= attrs:
                return True
        return NotImplemented
