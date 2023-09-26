from itertools import product
import numpy as np
import statsmodels.api as statsmodels
from scipy.optimize import curve_fit


def temporal_auto_correlate(maps, window_size):
    """
    Calculates the temporal auto correlation of each pixel with itself over different time lags.
    """
    stacked_maps = np.dstack(maps)
    nlags = int(min(10 * np.log10(stacked_maps.shape[2]), stacked_maps.shape[2] - 1))
    temporal_auto_correlations = np.zeros(
        shape=(stacked_maps.shape[0] - window_size + 1, stacked_maps.shape[1] - window_size + 1, nlags + 1))
    for x, y in product(range(temporal_auto_correlations.shape[0]), range(temporal_auto_correlations.shape[1])):
        vec = np.median(stacked_maps[x:x + window_size, y:y + window_size, :], axis=(0, 1))
        if np.all(vec == vec[0]):
            temporal_auto_correlations[x, y, :] = np.ones(shape=nlags + 1)
        else:
            temporal_auto_correlations[x, y, :] = statsmodels.tsa.stattools.acf(vec)
    return temporal_auto_correlations


# noinspection PyTupleAssignmentBalance
def calculate_taus(acorrs):
    """Given a 3d array where z vectors are auto correlations values in lags 0,...,acorrs.shape[2] fits the points to
    an exponential decay function and return an array of size (acorrs.shape[0],acorrs.shape[1]), with the fitted tau
    values."""

    def model_func(t, A, tau, C):
        return np.exp(-(1 / tau) * t) + C

    xdata = [i for i in range(acorrs.shape[2])]
    taus = np.zeros(shape=(acorrs.shape[0], acorrs.shape[1]))
    for x, y in product(range(taus.shape[0]), range(taus.shape[1])):
        # noinspection PyTupleAssignmentBalance
        # if np.all(acorrs[x, y, :] == acorrs[x, y, :][0]):
        #     taus[x, y] = 10
        #     continue
        # todo temp
        if np.all(acorrs[x, y, :] == acorrs[x, y, :][0]):
            taus[x, y] = -1
            continue
        opt_params, param_cov = curve_fit(f=model_func, xdata=xdata, ydata=acorrs[x, y, :],
                                          full_output=False, p0=(1.0, 1.0, 1.0), maxfev=5000)
        A, tau, C = opt_params
        taus[x, y] = tau
        if tau > 10:  # todo temp
            taus[x, y] = -1
    return taus
