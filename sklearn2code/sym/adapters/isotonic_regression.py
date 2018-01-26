from sympy.core.numbers import RealNumber
from sympy.functions.elementary.piecewise import Piecewise
from sympy.core.symbol import Symbol
from sklearn.isotonic import IsotonicRegression
import numpy as np
from ..sympy_special_values import NAN
from ..base import syms, sym_predict
from ..function import Function

@syms.register(IsotonicRegression)
def syms_isotonic_regression(estimator):
    return (Symbol('x'),)

def sym_linear_interp(variable, lower_x, upper_x, lower_y, upper_y):
    slope = RealNumber((upper_y - lower_y) / (upper_x - lower_x))
    return slope * (variable - RealNumber(lower_x)) + RealNumber(lower_y) 

@sym_predict.register(IsotonicRegression)
def sym_predict_isotonic_regression(estimator):
    variable = syms(estimator)[0]
    pieces = []
    try:
        x_upper = estimator.f_.x[0]
        y_upper = estimator.f_.y[0]
    except AttributeError:
        return RealNumber(estimator.f_(np.array([0.]))[0])
    i = 0
    n = len(estimator.f_.x)
    if estimator.out_of_bounds == 'clip':
        pieces.append((y_upper, variable < RealNumber(x_upper)))
    elif estimator.out_of_bounds == 'nan':
        pieces.append((NAN(1), variable < RealNumber(x_upper)))
    else:
        raise ValueError('out_of_bounds=%s not supported.' % estimator.out_of_bounds)
    
    while i < (n-1):
        i += 1
        x_lower = x_upper
        y_lower = y_upper
        x_upper = estimator.f_.x[i]
        y_upper = estimator.f_.y[i]
        pieces.append((sym_linear_interp(variable, x_lower, x_upper, y_lower, y_upper), (RealNumber(x_lower) <= variable) & (variable <= RealNumber(x_upper))))
    
    if estimator.out_of_bounds == 'clip':
        pieces.append((y_upper, variable >= RealNumber(x_upper)))
    elif estimator.out_of_bounds == 'nan':
        pieces.append((NAN(1), variable >= RealNumber(x_upper)))
    else:
        raise ValueError('out_of_bounds=%s not supported.' % estimator.out_of_bounds)
    
    return Function(syms(estimator), tuple(), Piecewise(*pieces))
