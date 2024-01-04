from itertools import product
import numpy as np
import statsmodels.api as statsmodels
from scipy.optimize import curve_fit


def temporal_auto_correlate(maps: list[np.ndarray], window_size: int = 1, nlags=None, max_freq=0.9):
    """
    Calculates the temporal auto correlation of each pixel with itself over different time lags.
    """
    stacked_maps = np.dstack(maps)
    # stacked_maps = scipy.ndimage.gaussian_filter(stacked_maps, sigma=1,
    #                                              radius=2, axes=[2])
    if not nlags:
        nlags = int(min(10 * np.log10(stacked_maps.shape[2]), stacked_maps.shape[2] - 1)) + 1
    temporal_auto_correlations = np.zeros(
        shape=(stacked_maps.shape[0] - window_size + 1, stacked_maps.shape[1] - window_size + 1, nlags))
    for x, y in product(range(temporal_auto_correlations.shape[0]), range(temporal_auto_correlations.shape[1])):
        vec = np.median(stacked_maps[x:x + window_size, y:y + window_size, :], axis=(0, 1))
        values, counts = np.unique(vec, return_counts=True)
        most_frequent_value = values[counts.argmax()]
        frequency = counts.max() / len(vec)
        if frequency >= max_freq:
            temporal_auto_correlations[x, y, :] = np.ones(shape=nlags)
        else:
            temporal_auto_correlations[x, y, :] = statsmodels.tsa.stattools.acf(vec, nlags=nlags - 1)
    return temporal_auto_correlations


def exponent_model_func(t: float, tau: float):
    return np.exp(-(1 / tau) * t)


def double_exponent_model_func(t: float, tau_u: float, tau_f: float, s_u: float, s_f: float):
    c_u = 1 - (s_u ** 2)
    c_f = (s_u ** 2) * (1 - s_f ** 2)
    s_md = (s_u ** 2) * (s_f ** 2)
    constraint_indicators = [tau_u < 0,
                             tau_f < tau_u,
                             s_u ** 2 > 1,
                             s_f ** 2 > 1]
    return s_md ** 2 + c_u * np.exp(-(t / tau_u)) + c_f * np.exp(-(t / tau_f)) + np.sum(constraint_indicators) * 9999999


# noinspection PyTupleAssignmentBalance
def calculate_taus(acorrs: np.ndarray):
    """Given a 3d array where z vectors are auto correlations values in lags 0,...,acorrs.shape[2] fits the points to
    an exponential decay function and return an array of size (acorrs.shape[0],acorrs.shape[1]), with the fitted tau
    values."""

    xdata = [i for i in range(acorrs.shape[2])]
    taus = np.zeros(shape=(acorrs.shape[0], acorrs.shape[1]))
    for x, y in product(range(taus.shape[0]), range(taus.shape[1])):
        if np.all(acorrs[x, y, :] == acorrs[x, y, :][0]):
            taus[x, y] = -1
            continue
        try:
            tau, param_cov = curve_fit(f=model_func, xdata=xdata, ydata=acorrs[x, y, :],
                                       full_output=False, p0=(1.0), maxfev=5000)
        except RuntimeError:
            tau = -1
        taus[x, y] = tau
    return taus
