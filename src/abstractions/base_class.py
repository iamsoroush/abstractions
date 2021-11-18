from abc import ABC, abstractmethod

from .utils import ConfigStruct


class BaseClass(ABC):

    def __init__(self, config: ConfigStruct = None):
        self._set_defaults()
        self.config = config
        if config is not None:
            self._load_params(config)

    @abstractmethod
    def _load_params(self, config):
        """Read parameters from config file."""

        pass

    @abstractmethod
    def _set_defaults(self):
        """Default values for your class, if ``None`` is passed as config.

        Should initialize the same parameters as in ``_load_params`` method.
        """

        pass
