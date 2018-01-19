from numpy import equal, where, isnan, maximum, minimum, exp, logical_not, logical_and, logical_or, select, less_equal, greater_equal, less, greater, nan, inf, log
from scipy.special import expit
%for function in functions:
def ${namer(function)}(${', '.join(map(str, function.inputs))}):
%for assignments, (called_function, arguments) in function.calls:
    ${', '.join(map(str, assignments))} = ${namer(called_function)}(${', '.join(map(str, arguments))})
%endfor
    return ${', '.join(map(printer, function.outputs))}
%endfor



