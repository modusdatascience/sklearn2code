from sklearn.linear_model.logistic import LogisticRegression
from ..dispatching import call_method_or_dispatch, create_registerer

def input_size_from_coef(estimator):
    coef = estimator.coef_
    n_inputs = coef.shape[-1]
    return n_inputs

def input_size_from_n_features_(estimator):
    return estimator.n_features_

def input_size_from_n_features(estimator):
    return estimator.n_features

input_size_dispatcher = {
                         }

input_size = call_method_or_dispatch('input_size', input_size_dispatcher)
register_input_size = create_registerer(input_size_dispatcher, 'register_input_size')
