from ..input_size import register_input_size
from pyearth.earth import Earth
from pyearth.export import export_sympy, export_sympy_term_expressions
from ..sym_predict import register_sym_predict
from ..sym_transform import register_sym_transform
from ..syms import register_syms
from sympy.core.symbol import Symbol

def input_size_earth(estimator):
    return len(estimator.xlabels_)

def syms_earth(estimator):
    return [Symbol(label) for label in estimator.xlabels_]

register_input_size(Earth, input_size_earth)
register_sym_predict(Earth, export_sympy)
register_sym_transform(Earth, export_sympy_term_expressions)
register_syms(Earth, syms_earth)