from numpy import vectorize, array, bincount, argmax, apply_along_axis, array, argmax, bincount, equal, where, isnan, maximum, minimum, exp, logical_not, logical_and, logical_or, select, less_equal, greater_equal, less, greater, nan, inf, log
from scipy.special import expit
from toolz import compose
from functools import partial

def weighted_mode(*args):
    data, weights = zip(args)
    data = array(data)
    weights = array(weights)
    return apply_along_axis(lambda x: argmax(
                     bincount(x, weights=weights)),
                     axis=1, arr=data)

%for function in functions:
def ${namer(function)}(${', '.join(map(str, function.inputs)) + ', ' if function.inputs else ''}**kwargs):
%for assignments, (called_function, arguments) in function.calls:
    ${', '.join(map(str, assignments))} = ${namer(called_function)}(${', '.join(map(str, arguments))})
%endfor
    return ${', '.join(map(printer, function.outputs))}
%endfor



