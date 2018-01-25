from sympy.core.numbers import RealNumber, Zero, One
from sklearn.ensemble.gradient_boosting import BinomialDeviance,\
    LogOddsEstimator, GradientBoostingClassifier, QuantileEstimator,\
    LossFunction, MeanEstimator, ZeroEstimator, GradientBoostingRegressor,\
    BaseGradientBoosting, PriorProbabilityEstimator
from operator import add
from sympy.core.symbol import Symbol
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import true
from distutils.version import LooseVersion
import sklearn
from six.moves import reduce
from ..sympy_special_values import Expit
from ..base import register_sym_decision_function, sym_predict,\
    register_sym_predict_proba, sym_decision_function, sym_score_to_proba,\
    register_sym_predict, input_size_from_n_features,\
    input_size_from_n_features_, register_input_size,\
    register_sym_score_to_proba, register_sym_score_to_decision
from ..function import Function
from ..base import VariableFactory, syms, register_syms
from ..function import cart

# def sym_log_odds_estimator_predict(estimator):
#     return RealNumber(estimator.prior)
# 
# sym_init_function_dispatcher = {LogOddsEstimator: sym_log_odds_estimator_predict}
# sym_init_function = call_method_or_dispatch('sym_init_function', sym_init_function_dispatcher)

@register_sym_decision_function(BaseGradientBoosting)
def sym_decision_function_base_gradient_boosting(estimator):
    learning_rate = RealNumber(estimator.learning_rate)
    n_classes = estimator.estimators_.shape[1]
    trees = [list(map(sym_predict, estimator.estimators_[:,i])) for i in range(n_classes)]
    tree_part = cart(*(learning_rate * reduce(add, trees[i]) for i in range(n_classes)))
    init_part = sym_predict(estimator.init_).concat_inputs(tree_part)
#     if not isinstance(init_part, list):
#         init_part = [init_part]
    return tree_part + init_part
#     result = [tree_part[i] + init_part[i] for i in range(n_classes)]
#     return Function(syms(estimator), tuple(), tuple(result))
    
@register_sym_predict_proba(GradientBoostingClassifier)
def sym_predict_proba_gradient_boosting_classifier(estimator):
    score = sym_decision_function(estimator)
    score_to_proba = sym_score_to_proba(estimator.loss_)
    return score_to_proba.compose(score)
#     score_to_proba_syms = syms(estimator.loss_)
#     return tuple(score_to_proba_expr.subs({score_to_proba_syms[0]: score}) for score_to_proba_expr in score_to_proba_exprs)

@register_sym_predict(GradientBoostingRegressor)
def sym_predict_gradient_boosting_regressor(estimator):
    return sym_decision_function(estimator)

register_input_size(BaseGradientBoosting, input_size_from_n_features if LooseVersion(sklearn.__version__) < LooseVersion('0.19') else input_size_from_n_features_)

@register_sym_predict(PriorProbabilityEstimator)
def sym_predict_prior_probability_estimator(estimator):
    return Function(syms(estimator), tuple(), tuple(map(RealNumber, estimator.priors)))
#     result = map(RealNumber, estimator.priors)
#     
#     
#     if len(result) == 1:
#         return result[0]
#     else:
#         return result

@register_sym_predict(QuantileEstimator)
def sym_predict_quantile_estimator(estimator):
    return Function(syms(estimator), tuple(), (RealNumber(estimator.quantile),))


@register_sym_predict(LogOddsEstimator)
def sym_predict_log_odds_estimator(estimator):
    return Function(syms(estimator), tuple(), (RealNumber(estimator.prior),))


@register_sym_predict(MeanEstimator)
def sym_predict_mean_estimator(estimator):
    return Function(tuple(), tuple(), (RealNumber(estimator.mean),))

@register_sym_predict(ZeroEstimator)
def sym_predict_zero_estimator(estimator):
    return Function(tuple(), tuple(), (Zero(),))

@register_syms(LossFunction)
def syms_loss_function(loss):
    return (Symbol('x'),)

@register_sym_score_to_proba(BinomialDeviance)
def sym_score_to_proba_binomial_deviance(loss):
    inputs = syms(loss)
    calls = tuple()
    outputs = (1 - Expit(inputs[0]), Expit(inputs[0]))
    return Function(inputs, calls, outputs)

@register_sym_score_to_decision(BinomialDeviance)
def sym_score_to_decision_binomial_deviance(loss):
    score_to_proba = sym_score_to_proba(loss)
    Var = VariableFactory()
    compl_proba, proba = (Var(), Var())
    inputs = (compl_proba, proba)
    calls = tuple()
    outputs = (Piecewise((One(), proba > compl_proba), (Zero(), True)),)
    return Function(inputs, calls, outputs).compose(score_to_proba)



