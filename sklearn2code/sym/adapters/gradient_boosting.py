from sympy.core.numbers import RealNumber, Zero
from sklearn.ensemble.gradient_boosting import BinomialDeviance,\
    LogOddsEstimator, GradientBoostingClassifier, QuantileEstimator,\
    LossFunction, MeanEstimator, ZeroEstimator, GradientBoostingRegressor,\
    BaseGradientBoosting, PriorProbabilityEstimator
from operator import add
from ..sym_predict_proba import register_sym_predict_proba
from ..sym_predict import sym_predict
from ..input_size import register_input_size,\
    input_size_from_n_features, input_size_from_n_features_
from sympy.core.symbol import Symbol
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import true
from distutils.version import LooseVersion
import sklearn
from six.moves import reduce
from ..sym_decision_function import register_sym_decision_function,\
    sym_decision_function
from ..sym_score_to_proba import sym_score_to_proba,\
    register_sym_score_to_proba
from ..syms import syms, register_syms
from ..sym_predict import register_sym_predict
from ..sympy_special_values import Expit
from ..sym_score_to_decision import register_sym_score_to_decision

# def sym_log_odds_estimator_predict(estimator):
#     return RealNumber(estimator.prior)
# 
# sym_init_function_dispatcher = {LogOddsEstimator: sym_log_odds_estimator_predict}
# sym_init_function = call_method_or_dispatch('sym_init_function', sym_init_function_dispatcher)

@register_sym_decision_function(BaseGradientBoosting)
def sym_decision_function_gradient_boosting_classifier(estimator):
    learning_rate = RealNumber(estimator.learning_rate)
    n_classes = estimator.estimators_.shape[1]
    trees = [list(map(sym_predict, estimator.estimators_[:,i])) for i in range(n_classes)]
    tree_part = [learning_rate * reduce(add, trees[i]) for i in range(n_classes)]
    init_part = sym_predict(estimator.init_)
    if not isinstance(init_part, list):
        init_part = [init_part]
    result = [tree_part[i] + init_part[i] for i in range(n_classes)]
    if len(result) == 1:
        return result[0]
    else:
        return result
    
@register_sym_predict_proba(GradientBoostingClassifier)
def sym_predict_proba_gradient_boosting_classifier(estimator):
    score = sym_decision_function(estimator)
    score_to_proba_expr = sym_score_to_proba(estimator.loss_)
    score_to_proba_syms = syms(estimator.loss_)
    assert len(score_to_proba_syms) == 1
    return score_to_proba_expr.subs({score_to_proba_syms[0]: score})

@register_sym_predict(GradientBoostingRegressor)
def sym_predict_gradient_boosting_regressor(estimator):
    return sym_decision_function(estimator)

register_input_size(BaseGradientBoosting, input_size_from_n_features if LooseVersion(sklearn.__version__) < LooseVersion('0.19') else input_size_from_n_features_)

@register_sym_predict(PriorProbabilityEstimator)
def sym_predict_prior_probability_estimator(estimator):
    result = map(RealNumber, estimator.priors)
    if len(result) == 1:
        return result[0]
    else:
        return result

@register_sym_predict(QuantileEstimator)
def sym_predict_quantile_estimator(estimator):
    return RealNumber(estimator.quantile)

@register_sym_predict(LogOddsEstimator)
def sym_predict_log_odds_estimator(estimator):
    return RealNumber(estimator.prior)

@register_sym_predict(MeanEstimator)
def sym_predict_mean_estimator(estimator):
    return RealNumber(estimator.mean)

@register_sym_predict(ZeroEstimator)
def sym_predict_zero_estimator(estimator):
    return Zero()

@register_syms(LossFunction)
def syms_loss_function(loss):
    return [Symbol('x')]

@register_sym_score_to_proba(BinomialDeviance)
def sym_score_to_proba_binomial_deviance(loss):
    symbols = syms(loss)
    assert len(symbols) == 1
    return Expit(symbols[0])

@register_sym_score_to_decision(BinomialDeviance)
def sym_score_to_decision(loss):
    return Piecewise((RealNumber(1), sym_score_to_proba(loss) > RealNumber(1)/RealNumber(2)), (RealNumber(0), true))



