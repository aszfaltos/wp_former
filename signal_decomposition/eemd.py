import numpy as np
from PyEMD import EEMD as PY_EEMD


class EEMDWrapper:
    def __init__(self, data: np.ndarray, imfs=-1, random_seed=42, spline_kind='akima'):
        self._original = data
        self._eemd = PY_EEMD(spline_kind=spline_kind)
        self._eemd.noise_seed(random_seed)
        self._eemd.eemd(data, max_imf=imfs)
        self._imfs, self._residue = self._eemd.get_imfs_and_residue()

    def get_imfs(self):
        return self._imfs

    def get_residue(self):
        return self._residue

    def get_original(self):
        return self._original

    @staticmethod
    def reconstruct(imfs: np.ndarray, residue: np.ndarray) -> np.ndarray:
        return imfs.sum(0) + residue
