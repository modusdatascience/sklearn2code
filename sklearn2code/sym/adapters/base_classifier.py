from sklearn.base import ClassifierMixin
from sklearn2code.sym.base import sym_predict, sym_predict_proba
from sklearn2code.sym.expression import RealVariable, Integer, Piecewise,\
    RealNumber, true
from sklearn2code.sym.function import Function


@sym_predict.register(ClassifierMixin)
def sym_predict_classifier(estimator):
    x = RealVariable('x')
    return Function.from_expression(Piecewise((Integer(1), x >= RealNumber(.5)), (Integer(0), true))).compose(sym_predict_proba(estimator).select_outputs(1))
