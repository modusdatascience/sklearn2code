from pyearth.earth import Earth
from pyearth.export import export_sympy, export_sympy_term_expressions
from ..syms import register_syms
from sympy.core.symbol import Symbol
from ..base import register_input_size, register_sym_predict,\
    register_sym_transform
    
@register_input_size(Earth)
def input_size_earth(estimator):
    return len(estimator.xlabels_)

@register_syms(Earth)
def syms_earth(estimator):
    return [Symbol(label) for label in estimator.xlabels_]

register_sym_predict(Earth, export_sympy)
register_sym_transform(Earth, export_sympy_term_expressions)
