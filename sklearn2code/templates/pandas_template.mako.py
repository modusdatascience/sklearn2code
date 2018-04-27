from numpy import arange, newaxis, cumsum, vectorize, array, bincount, argmax, apply_along_axis, asarray, array, argmax, argsort, transpose, bincount, equal, where, isnan, maximum, minimum, exp, logical_not, logical_and, logical_or, select, less_equal, greater_equal, less, greater, nan, inf, log
from scipy.special import expit
from pandas import DataFrame
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

<%!
from toolz import flip
%>
%for function in functions:
def ${namer(function)}(dataframe):
    dataframe = dataframe.copy(deep=False)
%for assignments, (called_function, arguments) in function.calls:
    dataframe[[${', '.join(map(lambda x: '\'%s\'' % str(x), assignments))}]] = ${namer(called_function)}(dataframe[[${', '.join(map(lambda x: '\'%s\'' % str(x), arguments))}]].rename(columns=${repr(dict(zip(map(flip(getattr)('name'), arguments), map(flip(getattr)('name'), called_function.inputs))))}, copy=False))
%endfor
    result = DataFrame(index=dataframe.index)
% for i, output in enumerate(function.outputs):
    result[${i}] = ${printer(output, 0)}
%endfor
    return result
%endfor
