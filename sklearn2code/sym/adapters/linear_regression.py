from sympy.core.numbers import RealNumber
from ..syms import syms
import numpy as np
from sklearn.linear_model.base import LinearRegression
from ..sym_predict import register_sym_predict

@register_sym_predict(LinearRegression)
def sym_predict_linear(estimator):
    if hasattr(estimator, 'intercept_'):
        expression = RealNumber(estimator.intercept_[0])
    else:
        expression = RealNumber(0)
    symbols = syms(estimator)
    for coef, sym in zip(np.ravel(estimator.coef_), symbols):
        expression += RealNumber(coef) * sym
    return expression

