from numpy import ravel, stack, arange, newaxis, cumsum, vectorize, array, bincount, argmax, apply_along_axis, asarray, array, argmax, argsort, transpose, bincount, equal, where, isnan, maximum, minimum, exp, logical_not, logical_and, logical_or, select, less_equal, greater_equal, less, greater, nan, inf, log
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

def ordered(data):
    arr = stack(map(ravel,data), axis=1)
    for i in range(arr.shape[0]):
        arr[i,:] = np_sort(arr[i,:])
    return tuple(arr[:, i] for i in range(arr.shape[1]))


def normalize_l1(data):
    norm = sum(np.abs(x) for x in data)
    return tuple(np.where(norm != 0, x / norm, x) for x in data)

%for function in functions:
def ${namer(function)}(kwargs):
%for input in function.inputs:
    ${str(input)} = kwargs['${str(input)}']
%endfor

%for assignments, (called_function, arguments) in function.calls:
    ${', '.join(map(str, assignments))} = ${namer(called_function)}({${', '.join(map(lambda x: '"%s": %s' % (str(x[0]), str(x[1])), zip(called_function.inputs, arguments)))}})
%endfor
    return ${', '.join(map(printer, function.outputs))}

%endfor




