from sklearn.ensemble.weight_boosting import AdaBoostRegressor
from sklearn2code.sym.base import sym_predict, syms, sym_inverse_transform
import numpy as np
from sklearn2code.sym.function import Function, comp
from sklearn2code.sym.expression import WeightedMedian


# TODO: naming
def sym_weighted_median(preds, weights):
    return WeightedMedian(preds, weights)

@sym_predict.register(AdaBoostRegressor)
def sym_predict_ada_boost_regressor(estimator):
    estimators = estimator.estimators_
    weights = estimator.estimator_weights_
    preds = [sym_predict(est) for est in estimators]
    return Function(inputs=preds, calls=tuple(), outputs=sym_weighted_median(preds, weights))
    



    
    
