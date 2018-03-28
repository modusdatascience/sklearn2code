from sklearn.ensemble.weight_boosting import AdaBoostRegressor
from sklearn2code.sym.base import sym_predict, syms, sym_inverse_transform
import numpy as np
from sklearn2code.sym.function import Function, comp, cart, VariableFactory
from sklearn2code.sym.expression import WeightedMedian, RealNumber


def sym_weighted_median(preds, weights):
    return WeightedMedian(preds, weights)

@sym_predict.register(AdaBoostRegressor)
def sym_predict_ada_boost_regressor(estimator):
    estimators = estimator.estimators_
    weights = tuple(map(RealNumber, estimator.estimator_weights_))
    predictors = [sym_predict(est) for est in estimators]
    predictors = cart(*predictors)
    Var = VariableFactory()
    predictions = tuple(Var() for _ in predictors.outputs)

    return comp(Function(inputs=predictions, calls=tuple(), outputs=sym_weighted_median(predictions, weights)), 
                predictors) 