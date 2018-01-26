from .linear_regression import sym_predict_linear
from sklearn.linear_model.logistic import LogisticRegression
from ..sympy_special_values import Expit
from ..function import Function
from ..base import VariableFactory, sym_predict_proba, input_size_from_coef, input_size
from sympy.core.numbers import One

@sym_predict_proba.register(LogisticRegression)
def sym_predict_proba_logistic_regression(estimator):
    linear_part = sym_predict_linear(estimator)
    Var = VariableFactory()
    x = Var()
    return Function((x,), tuple(), (One() - Expit(x), Expit(x))).compose(linear_part)

input_size.register(LogisticRegression, input_size_from_coef)
