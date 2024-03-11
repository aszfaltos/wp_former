import numpy as np
from PyEMD import EEMD as PY_EEMD
from .preprocessor import Preprocessor


class EEMDPreprocessor(Preprocessor):
    def __init__(self, imfs=-1, random_seed=42, spline_kind='akima', trials=100):
        self._eemd = PY_EEMD(spline_kind=spline_kind, parallel=True, trials=trials)
        self._eemd.noise_seed(random_seed)
        self._imfs = imfs

    @staticmethod
    def reconstruct(imfs: np.ndarray, residue: np.ndarray) -> np.ndarray:
        return imfs.sum(0) + residue

    def process(self, data: np.ndarray) -> np.ndarray:
        self._eemd.eemd(data, max_imf=self._imfs, progress=True)
        imfs, residue = self._eemd.get_imfs_and_residue()
        return np.concatenate([imfs.T, residue[..., np.newaxis]], dtype=np.float32, axis=1)
