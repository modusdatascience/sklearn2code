from numpy import vectorize, array, bincount, argmax, apply_along_axis, equal, where, isnan, maximum, minimum, exp, logical_not, logical_and, logical_or, select, less_equal, greater_equal, less, greater, nan, inf, log, asarray
from scipy.special import expit
from pandas import DataFrame
from toolz import compose
from functools import partial

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
    result[${i}] = ${printer(output)}
%endfor
    return result
%endfor
