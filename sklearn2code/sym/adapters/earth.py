from pyearth.earth import Earth
from pyearth.export import export_sympy, export_sympy_term_expressions
from sympy.core.symbol import Symbol
from ..base import register_input_size, register_sym_predict,\
    register_sym_transform
from ..base import register_syms
from ..base import syms
from ..function import Function, tupify

@register_input_size(Earth)
def input_size_earth(estimator):
    return len(estimator.xlabels_)

@register_syms(Earth)
def syms_earth(estimator):
    return [Symbol(label) for label in estimator.xlabels_]

@register_sym_transform(Earth)
def sym_transform_earth(estimator):
    inputs = syms(estimator)
    calls = tuple()
    outputs = tuple(export_sympy_term_expressions(estimator))
    return Function(inputs, calls, outputs)

@register_sym_predict(Earth)
def sym_predict_earth(estimator):
    inputs = syms(estimator)
    calls = tuple()
    outputs = tupify(export_sympy(estimator))
    return Function(inputs, calls, outputs)
