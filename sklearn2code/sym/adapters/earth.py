from pyearth.earth import Earth
from pyearth.export import export_sympy, export_sympy_term_expressions
from sympy.core.symbol import Symbol
from ..base import syms
from ..function import Function, tupify
from sklearn2code.sym.base import sym_predict, sym_transform, input_size

@input_size.register(Earth)
def input_size_earth(estimator):
    return len(estimator.xlabels_)

@syms.register(Earth)
def syms_earth(estimator):
    return [Symbol(label) for label in estimator.xlabels_]

@sym_transform.register(Earth)
def sym_transform_earth(estimator):
    inputs = syms(estimator)
    calls = tuple()
    outputs = tuple(export_sympy_term_expressions(estimator))
    return Function(inputs, calls, outputs)

@sym_predict.register(Earth)
def sym_predict_earth(estimator):
    inputs = syms(estimator)
    calls = tuple()
    outputs = tupify(export_sympy(estimator))
    return Function(inputs, calls, outputs)
