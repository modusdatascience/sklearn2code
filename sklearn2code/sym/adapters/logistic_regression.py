from .linear_regression import sym_predict_linear
from sklearn.linear_model.logistic import LogisticRegression
from ..sympy_special_values import Expit
from ..base import register_input_size, register_sym_predict,\
    input_size_from_coef, register_sym_predict_proba
from ..function import Function
from sklearn2code.sym.base import VariableFactory
from sympy.core.numbers import One

def sym_predict_proba_logistic_regression(estimator):
    linear_part = sym_predict_linear(estimator)
    Var = VariableFactory()
    x = Var()
    return Function((x,), tuple(), (One() - Expit(x), Expit(x))).compose(linear_part)

def sym_predict_logistic_regression(estimator):
    return sym_predict_proba_logistic_regression(estimator).select_outputs(1)
    
    

register_sym_predict(LogisticRegression, sym_predict_logistic_regression)
register_input_size(LogisticRegression, input_size_from_coef)
register_sym_predict_proba(LogisticRegression, sym_predict_proba_logistic_regression)