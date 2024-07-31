from numpy import ndarray
import numpy as np
import pywt
from .modwt import modwt, modwtmra, imodwt
from .preprocessor import Preprocessor


class WaveletPreprocessor(Preprocessor):
    def __init__(self, features: int, wavelet: str, decomposition_lvl: int):
        self.features = features
        self._decomposition_lvl = decomposition_lvl
        self._wavelet = wavelet

    def reconstruct(self, coefs: list[ndarray], wavelet: str):
        features = []
        for i in range(0, self.features, self._decomposition_lvl):
            features.append(imodwt(coefs[i:i+self._decomposition_lvl], wavelet))
        return features

    def reconstruct_mra(self, coefs: np.ndarray):
        features = []
        for i in range(0, self.features, self._decomposition_lvl):
            features.append(coefs[i:i + self._decomposition_lvl].sum(-1))
        return np.stack(features, axis=-1)

    def process(self, data: ndarray):
        decomposition_lvl = self._decomposition_lvl
        max_lvl = pywt.dwt_max_level(len(data), self._wavelet)
        if decomposition_lvl > max_lvl:
            raise ValueError(f"Decomposition level {decomposition_lvl} is higher than the maximum level {max_lvl}")
        mras = []
        for feature_idx in range(data.shape[-1]):
            coefs = modwt(data[..., feature_idx],  self._wavelet, level=decomposition_lvl)
            mras.append(np.array(modwtmra(coefs, self._wavelet), dtype=np.float32).T)
        return np.stack(mras, axis=-1).reshape(-1, 1, (decomposition_lvl + 1) * len(mras))
