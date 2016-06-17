# Code based on statsmodels (http://statsmodels.sourceforge.net)

import numpy as np

def acovf(x, unbiased=False, demean=True):
    '''
    Autocovariance for 1D

    Parameters
    ----------
    x : array
        Time series data. Must be 1d.
    unbiased : bool
        If True, then denominators is n-k, otherwise n
    demean : bool
        If True, then subtract the mean x from each element of x

    Returns
    -------
    acovf : array
        autocovariance function
    '''
    x = np.squeeze(np.asarray(x))
    if x.ndim > 1:
        raise ValueError("x must be 1d. Got %d dims." % x.ndim)
    n = len(x)

    if demean:
        xo = x - x.mean()
    else:
        xo = x
    if unbiased:
        xi = np.arange(1, n + 1)
        d = np.hstack((xi, xi[:-1][::-1]))
    else:
        d = n * np.ones(2 * n - 1)

    return (np.correlate(xo, xo, 'full') / d)[n - 1:]


#NOTE: Changed unbiased to False
#see for example
# http://www.itl.nist.gov/div898/handbook/eda/section3/autocopl.htm
def acf(x, unbiased=False, nlags=1):
    '''
    Autocorrelation function for 1d arrays.

    Parameters
    ----------
    x : array
       Time series data
    unbiased : bool
       If True, then denominators for autocovariance are n-k, otherwise n
    nlags: int, optional
        Number of lags to return autocorrelation for.

    Returns
    -------
    acf : array
        autocorrelation function

    Notes
    -----
    The acf at lag 0 (ie., 1) is returned.

    This is based np.correlate which does full convolution. For very long time
    series it is recommended to use fft convolution instead.

    If unbiased is true, the denominator for the autocovariance is adjusted
    but the autocorrelation is not an unbiased estimtor.
    '''
    avf = acovf(x, unbiased=unbiased, demean=True)
    #acf = np.take(avf/avf[0], range(1,nlags+1))
    acf = avf[:nlags + 1] / avf[0]

    return acf
