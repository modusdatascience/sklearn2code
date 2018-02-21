from sklearn.isotonic import IsotonicRegression
import numpy as np
from ..base import syms, sym_predict
from ..function import Function
from ..expression import RealVariable, RealNumber, Piecewise, nan

@syms.register(IsotonicRegression)
def syms_isotonic_regression(estimator):
    return (RealVariable('x'),)

def sym_linear_interp(variable, lower_x, upper_x, lower_y, upper_y):
    slope = (upper_y - lower_y) / (upper_x - lower_x)
    return slope * (variable - lower_x) + lower_y

@sym_predict.register(IsotonicRegression)
def sym_predict_isotonic_regression(estimator):
    variable = syms(estimator)[0]
    pieces = []
    try:
        x_upper = RealNumber(estimator.f_.x[0])
        y_upper = RealNumber(estimator.f_.y[0])
    except AttributeError:
        return RealNumber(estimator.f_(np.array([0.]))[0])
    i = 0
    n = len(estimator.f_.x)
    if estimator.out_of_bounds == 'clip':
        pieces.append((y_upper, variable < x_upper))
    elif estimator.out_of_bounds == 'nan':
        pieces.append((nan, variable < RealNumber(x_upper)))
    else:
        raise ValueError('out_of_bounds=%s not supported.' % estimator.out_of_bounds)
    
    while i < (n-1):
        i += 1
        x_lower = x_upper
        y_lower = y_upper
        x_upper = RealNumber(estimator.f_.x[i])
        y_upper = RealNumber(estimator.f_.y[i])
        pieces.append((sym_linear_interp(variable, x_lower, x_upper, y_lower, y_upper), (x_lower <= variable) & (variable <= x_upper)))
    
    if estimator.out_of_bounds == 'clip':
        pieces.append((y_upper, variable >= x_upper))
    elif estimator.out_of_bounds == 'nan':
        pieces.append((nan, variable >= x_upper))
    else:
        raise ValueError('out_of_bounds=%s not supported.' % estimator.out_of_bounds)
    
    return Function(syms(estimator), tuple(), Piecewise(*pieces))
