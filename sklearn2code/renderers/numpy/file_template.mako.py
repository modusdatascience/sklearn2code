from numpy import arange, newaxis, cumsum, vectorize, array, bincount, argmax, apply_along_axis, asarray, array, argmax, argsort, transpose, bincount, equal, where, isnan, maximum, minimum, exp, logical_not, logical_and, logical_or, select, less_equal, greater_equal, less, greater, nan, inf, log
from scipy.special import expit
from toolz import compose
from functools import partial

def weighted_median(data, weights):
    data = data.T
    sorted_idx = argsort(data, axis=1)
    weight_cdf = cumsum(weights[sorted_idx], axis=1)
    median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, newaxis]
    median_idx = median_or_above.argmax(axis=1)
    medians = sorted_idx[arange(data.shape[0]), median_idx]
    return data[arange(data.shape[0]), medians]

%for rendered_function in rendered_functions:
${rendered_function}

%endfor