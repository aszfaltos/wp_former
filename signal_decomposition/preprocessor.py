from abc import ABC, abstractmethod
from numpy import ndarray
import numpy as np


class Preprocessor(ABC):
    @abstractmethod
    def process(self, x: ndarray) -> ndarray:
        pass


class SimplePreprocessor(Preprocessor):
    def process(self, x: ndarray) -> ndarray:
        return np.array(x[..., np.newaxis], dtype=np.float32)
