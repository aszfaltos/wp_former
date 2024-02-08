from numpy import ndarray
import numpy as np
import pywt
from .modwt import modwt, modwtmra, imodwt


class WaveletWrapper:
    def __init__(self, data: ndarray, wavelet: str, decomposition_lvl: int | None = None):
        self._original = data
        self._decomposition_lvl = decomposition_lvl
        if decomposition_lvl is None:
            self.decomposition_lvl = pywt.dwt_max_level(len(data), wavelet)

        self._wavelet = wavelet
        self._coefs = modwt(data, wavelet, level=decomposition_lvl)
        self._mra = modwtmra(self._coefs, wavelet)

    def get_original(self):
        return self._original

    def get_coefs(self):
        return self._coefs

    def get_mra(self):
        return self._mra

    @staticmethod
    def reconstruct(coefs: list[ndarray], wavelet: str):
        return imodwt(coefs, wavelet)

    @staticmethod
    def reconstruct_mra(coefs: np.ndarray):
        return coefs.sum(-1)
