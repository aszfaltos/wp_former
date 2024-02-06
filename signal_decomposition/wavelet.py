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
            return pywt.waverec(coefs, wavelet)

        reconstructed = []

        div_factor = 1
        for idx, coef in enumerate(reversed(coefs[1:])):
            div_factor = div_factor * 2
            sampled = coef[
                          WaveletWrapper._scaled_indexes(math.ceil(coef.shape[0] / div_factor), (coef.shape[0] - 1))
                      ][::1]
            reconstructed.insert(0, sampled)

        reconstructed.insert(0,
                             coefs[0][
                                 WaveletWrapper._scaled_indexes(math.ceil(coefs[0].shape[0] / div_factor),
                                                                (coefs[0].shape[0] - 1))
                             ][::1])

        return pywt.waverec(reconstructed, wavelet), reconstructed

    @staticmethod
    def _interpolate(coefs, spline, original_len):
        interpolated = []
        if spline == 'akima':
            for coef in coefs:
                s = scipy.interpolate.Akima1DInterpolator(np.arange(len(coef)), coef)
                interpolated.append(s(WaveletWrapper._sample_positions(original_len, len(coef))))
        elif spline == 'cubic':
            for coef in coefs:
                s = scipy.interpolate.CubicSpline(np.arange(len(coef)), coef)
                interpolated.append(s(WaveletWrapper._sample_positions(original_len, len(coef))))
        elif spline == 'linear':
            for coef in coefs:
                s = scipy.interpolate.interp1d(np.arange(len(coef)), coef)
                interpolated.append(s(WaveletWrapper._sample_positions(original_len, len(coef))))
        else:
            return coefs

        return interpolated

    @staticmethod
    def _scaled_indexes(num_of_indexes, target_ceil):
        return np.round(np.arange(num_of_indexes) / (num_of_indexes - 1) * target_ceil).astype(int)

    @staticmethod
    def _sample_positions(orig_len, coef_len):
        return ((np.arange(orig_len) / ((orig_len - 1) / (coef_len - 1)))
                .clip(0, coef_len - 1))
