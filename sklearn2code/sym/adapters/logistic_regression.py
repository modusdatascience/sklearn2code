from .linear_regression import sym_predict_linear
from sklearn.linear_model.logistic import LogisticRegression
from ..sympy_special_values import Expit
from ..base import register_input_size, register_sym_predict,\
    input_size_from_coef, register_sym_predict_proba

def sym_predict_logistic_regression(logistic_regression):
    return Expit(sym_predict_linear(logistic_regression))

register_sym_predict(LogisticRegression, sym_predict_logistic_regression)
register_input_size(LogisticRegression, input_size_from_coef)
register_sym_predict_proba(LogisticRegression, sym_predict_logistic_regression,)