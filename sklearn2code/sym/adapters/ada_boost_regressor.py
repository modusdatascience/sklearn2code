from sklearn.ensemble.weight_boosting import AdaBoostRegressor
from sklearn2code.sym.base import sym_predict
import numpy as np




@sym_predict.register(AdaBoostRegressor)
def sym_predict_ada_boost_regressor(estimator):
    estimators = estimator.estimators_
    weights = estimator.estimator_weights_
    
    ''' 
    See, this doesn't make sense to me. I need the full set of predictions (and not the function version) to pick a weighted median
    '''
    
    preds = np.array([sym_predict(est) for est in estimators])
    1 + 1
    # do we need to transpose? 
