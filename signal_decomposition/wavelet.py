from numpy import ndarray
import numpy as np
import pywt
import scipy
import math


class WaveletWrapper:
    def __init__(self, data: ndarray, wavelet: str, spline: str | None = None, decomposition_lvl: int | None = None):
        self._original = data
        self._decomposition_lvl = decomposition_lvl
        if decomposition_lvl is None:
            self.decomposition_lvl = pywt.dwt_max_level(len(data), wavelet)

        self._spline = spline
        self._wavelet = wavelet
        self._coefs = pywt.wavedec(data, wavelet, level=decomposition_lvl)
        self._interpolated = WaveletWrapper._interpolate(self._coefs, self._spline, len(self._original))

    def get_original(self):
        return self._original

    def get_coefs(self):
        return self._coefs

    def get_interpolated(self):
        return self._interpolated

    @staticmethod
    def reconstruct(coefs: list[ndarray], wavelet: str, interpolated=True):
        if not interpolated:
            for coef in coefs:
                print(coef.shape)
            return pywt.waverec(coefs, wavelet)

        reconstructed = []

        div_factor = 1
        for idx, coef in enumerate(reversed(coefs[1:])):
            div_factor = div_factor * 2
            sampled = coef[np.round((np.arange(math.ceil(coef.shape[0] / div_factor)) / (math.ceil(coef.shape[0] / div_factor) - 1) * (coef.shape[0] - 1))).astype(int)][::1]
            print(sampled.shape)
            reconstructed.insert(0, sampled)

        reconstructed.insert(0, coefs[0][np.round((np.arange(math.ceil(coefs[0].shape[0] / div_factor)) / (math.ceil(coefs[0].shape[0] / div_factor) - 1) * (coefs[0].shape[0] - 1))).astype(int)][::1])

        return pywt.waverec(reconstructed, wavelet), reconstructed

    @staticmethod
    def _interpolate(coefs, spline, original_len):
        interpolated = []
        if spline == 'akima':
            for coef in coefs:
                s = scipy.interpolate.Akima1DInterpolator(np.arange(len(coef)), coef)
                interpolated.append(s((np.arange(original_len) / ((original_len - 1) / (len(coef) - 1))).clip(0, len(coef) - 1)))
        elif spline == 'cubic':
            for coef in coefs:
                s = scipy.interpolate.CubicSpline(np.arange(len(coef)), coef)
                interpolated.append(s((np.arange(original_len) / ((original_len - 1) / (len(coef) - 1))).clip(0, len(coef) - 1)))
        elif spline == 'linear':
            for coef in coefs:
                s = scipy.interpolate.interp1d(np.arange(len(coef)), coef)
                interpolated.append(s((np.arange(original_len) / ((original_len - 1) / (len(coef) - 1))).clip(0, len(coef) - 1)))
        else:
            return coefs

        return interpolated

# TODO: clean up the indexing