from sklearn2code.sym.expression import RealNumber
from operator import __add__, __getitem__
from six.moves import reduce
from sklearn2code.sym.base import sym_predict, sym_predict_proba, syms
from sklearn.ensemble.bagging import BaggingRegressor, BaggingClassifier
from sklearn2code.sym.function import VariableFactory, Function
from toolz.functoolz import curry
from itertools import starmap

@sym_predict.register(BaggingRegressor)
def sym_predict_bagging_regressor(estimator):
    inputs = syms(estimator)
    Var = VariableFactory(existing=inputs)
    vars_ = tuple(Var() for _ in range(len(estimator.estimators_)))
    calls = tuple(starmap(lambda var, est, args: ((var,), (sym_predict(est), tuple(map(curry(__getitem__)(inputs), list(args))))), 
                zip(vars_, estimator.estimators_, estimator.estimators_features_)))
    outputs = (reduce(__add__, vars_) / RealNumber(len(estimator.estimators_)),)
    return Function(inputs, calls, outputs)

@sym_predict_proba.register(BaggingClassifier)
def sym_predict_proba_bagging_classifier(estimator):
    inputs = syms(estimator)
    Var = VariableFactory(existing=inputs)
    vars_ = tuple(Var() for _ in range(len(estimator.estimators_)))
    calls = tuple(starmap(lambda var, est, args: ((var,), (sym_predict_proba(est) if hasattr(est, 'predict_proba') else sym_predict(est), tuple(map(curry(__getitem__)(inputs), list(args))))), 
                zip(vars_, estimator.estimators_, estimator.estimators_features_)))
    outputs = (reduce(__add__, vars_) / RealNumber(len(estimator.estimators_)),)
    return Function(inputs, calls, outputs)

