from numpy import ndarray
import numpy as np
import pywt
from .modwt import modwt, modwtmra, imodwt
from .preprocessor import Preprocessor


class WaveletPreprocessor(Preprocessor):
    def __init__(self, wavelet: str, decomposition_lvl: int | None = None):
        self._decomposition_lvl = decomposition_lvl
        self._wavelet = wavelet

    @staticmethod
    def reconstruct(coefs: list[ndarray], wavelet: str):
        return imodwt(coefs, wavelet)

    @staticmethod
    def reconstruct_mra(coefs: np.ndarray):
        return coefs.sum(-1)

    def process(self, data: ndarray):
        decomposition_lvl = self._decomposition_lvl
        max_lvl = pywt.dwt_max_level(len(data), self._wavelet)
        if decomposition_lvl is None or decomposition_lvl > max_lvl:
            decomposition_lvl = max_lvl
        coefs = modwt(data,  self._wavelet, level=decomposition_lvl)
        mra = modwtmra(coefs, self._wavelet)
        return np.array(mra, dtype=np.float32).T
