from sklearn2code.sym.adapters.linear import sym_predict_linear
from sklearn.linear_model.logistic import LogisticRegression
from ..function import Function, VariableFactory
from ..base import sym_predict_proba, input_size_from_coef, input_size
from ..expression import Expit, RealNumber

@sym_predict_proba.register(LogisticRegression)
def sym_predict_proba_logistic_regression(estimator):
    linear_part = sym_predict_linear(estimator)
    Var = VariableFactory()
    x = Var()
    return Function((x,), tuple(), (RealNumber(1) - Expit(x), Expit(x))).compose(linear_part)

input_size.register(LogisticRegression, input_size_from_coef)
