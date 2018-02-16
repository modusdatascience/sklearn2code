from sympy.core.numbers import RealNumber
from ..base import syms
import numpy as np
from sklearn.linear_model.base import LinearRegression
from ..base import sym_predict
from ..function import Function
from sklearn.linear_model.coordinate_descent import Lasso

@sym_predict.register(Lasso)
@sym_predict.register(LinearRegression)
def sym_predict_linear(estimator):
    if hasattr(estimator, 'intercept_'):
        try:
            expression = RealNumber(estimator.intercept_[0])
        except IndexError: 
            expression = RealNumber(estimator.intercept_)
    else:
        expression = RealNumber(0)
    symbols = syms(estimator)
    for coef, sym in zip(np.ravel(estimator.coef_), symbols):
        expression += RealNumber(coef) * sym
    return Function(syms(estimator), tuple(), expression)

