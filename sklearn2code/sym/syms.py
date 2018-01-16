from sympy.core.symbol import Symbol
from sklearn.base import BaseEstimator
from ..dispatching import call_method_or_dispatch, create_registerer
from ..sym.base import input_size

def syms_x(estimator):
    return [Symbol('x%d' % d) for d in range(input_size(estimator))]

syms_dispatcher = {
                   BaseEstimator: syms_x,
                   }
syms = call_method_or_dispatch('syms', syms_dispatcher)
register_syms = create_registerer(syms_dispatcher, 'register_syms')