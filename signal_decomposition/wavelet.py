from numpy import ndarray
import numpy as np
import pywt
import scipy


# TODO: make it a class like eemd

def apply_wavelet(data: ndarray, wavelet: str, spline: str | None = None, decomp_lvl: int | None = None):
    if decomp_lvl is None:
        decomp_lvl = pywt.dwt_max_level(len(data), wavelet)

    coefs = pywt.wavedec(data, wavelet, level=decomp_lvl)

    interpolated = []
    if spline == 'akima':
        for coef in coefs:
            s = scipy.interpolate.Akima1DInterpolator(np.arange(0, len(coef)) * (len(data) / (len(coef) - 1)), coef)
            interpolated.append(s(np.arange(0, len(data))))
    elif spline == 'cubic':
        for coef in coefs:
            s = scipy.interpolate.CubicSpline(np.arange(0, len(coef)) * (len(data) / (len(coef) - 1)), coef)
            interpolated.append(s(np.arange(0, len(data))))
    elif spline == 'linear':
        for coef in coefs:
            s = scipy.interpolate.interp1d(np.arange(0, len(coef)) * (len(data) / (len(coef) - 1)), coef)
            interpolated.append(s(np.arange(0, len(data))))
    else:
        return coefs

    return interpolated
