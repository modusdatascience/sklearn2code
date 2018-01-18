import bisect
from nose.tools import assert_almost_equal
from sympy.core.numbers import RealNumber
from sympy.functions.elementary.piecewise import Piecewise
from sympy.core.symbol import Symbol
from sklearn.isotonic import IsotonicRegression
from numpy.testing.utils import assert_array_almost_equal
import numpy as np
from ..sympy_special_values import NAN
from ..base import register_sym_predict
from ..base import syms, register_syms

@register_syms(IsotonicRegression)
def syms_isotonic_regression(estimator):
    return [Symbol('x')]

def sym_linear_interp(variable, lower_x, upper_x, lower_y, upper_y):
    slope = RealNumber((upper_y - lower_y) / (upper_x - lower_x))
    return slope * (variable - RealNumber(lower_x)) + RealNumber(lower_y) 

@register_sym_predict(IsotonicRegression)
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
        pieces.append((NAN(), variable < RealNumber(x_upper)))
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
        pieces.append((NAN(), variable > RealNumber(x_upper)))
    else:
        raise ValueError('out_of_bounds=%s not supported.' % estimator.out_of_bounds)
    
    return Piecewise(*pieces)

# def predict_isotonic(estimator, value):
#     i = bisect.bisect(estimator.f_.x, value)
#     if i == 0:
#         return estimator.f_.y[0]
#     elif i>=len(estimator.f_.y):
#         return estimator.f_.y[-1]
#     else:
#         lower_y = estimator.f_.y[i-1]
#         upper_y = estimator.f_.y[i]
#         lower_x = estimator.f_.x[i-1]
#         upper_x = estimator.f_.x[i]
#         slope = (upper_y - lower_y) / (upper_x - lower_x)
#         return lower_y + slope * (value - lower_x)
#     
# if __name__ == '__main__':
#     from sklearntools.calibration import IsotonicRegressor
#     import numpy as np
#     X = np.random.normal(size=1000) + 100
#     y = np.random.normal(X ** 2, .1)
#     estimator = IsotonicRegressor(out_of_bounds='clip').fit(X, y)
#     for v in np.arange(-10,10,.1):
#         assert_almost_equal(predict_isotonic(estimator, v), estimator.predict([v])[0])
#     
#     code = model_to_code(estimator, 'numpy', 'predict', 'test_model')
#     numpy_test_module = exec_module('numpy_test_module', code)
#     y_pred_numpy = numpy_test_module.test_model(x=shrinkd(1, np.asarray(X)))
#     y_pred = estimator.predict(shrinkd(1,np.asarray(X)))
#     y_pred_test = [predict_isotonic(estimator, v) for v in X]
#     assert_array_almost_equal(np.ravel(y_pred_numpy), np.ravel(y_pred))
#     print('Success!')
#     