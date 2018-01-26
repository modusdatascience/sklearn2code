from sympy.core.numbers import RealNumber, Zero, One
from sklearn.ensemble.gradient_boosting import BinomialDeviance,\
    LogOddsEstimator, GradientBoostingClassifier, QuantileEstimator,\
    LossFunction, MeanEstimator, ZeroEstimator, GradientBoostingRegressor,\
    BaseGradientBoosting, PriorProbabilityEstimator, ClassificationLossFunction
from operator import add
from sympy.core.symbol import Symbol
from sympy.functions.elementary.piecewise import Piecewise
from distutils.version import LooseVersion
import sklearn
from six.moves import reduce
from ..sympy_special_values import Expit
from ..base import sym_predict,\
    sym_decision_function, sym_score_to_proba,\
    input_size_from_n_features,\
    input_size_from_n_features_, sym_score_to_decision
from ..function import Function
from ..base import VariableFactory, syms, sym_predict_proba, input_size
from ..function import cart

@sym_decision_function.register(BaseGradientBoosting)
def sym_decision_function_base_gradient_boosting(estimator):
    learning_rate = RealNumber(estimator.learning_rate)
    n_classes = estimator.estimators_.shape[1]
    trees = [list(map(sym_predict, estimator.estimators_[:,i])) for i in range(n_classes)]
    tree_part = cart(*(learning_rate * reduce(add, trees[i]) for i in range(n_classes)))
    init_part = sym_predict(estimator.init_).append_inputs(tree_part.inputs)
    return tree_part + init_part
    
@sym_predict_proba.register(GradientBoostingClassifier)
def sym_predict_proba_gradient_boosting_classifier(estimator):
    score = sym_decision_function(estimator)
    score_to_proba = sym_score_to_proba(estimator.loss_)
    return score_to_proba.compose(score)

@sym_predict.register(GradientBoostingRegressor)
def sym_predict_gradient_boosting_regressor(estimator):
    return sym_decision_function(estimator)

input_size.register(BaseGradientBoosting, input_size_from_n_features if LooseVersion(sklearn.__version__) < LooseVersion('0.19') else input_size_from_n_features_)

@sym_predict.register(PriorProbabilityEstimator)
def sym_predict_prior_probability_estimator(estimator):
    return Function(syms(estimator), tuple(), tuple(map(RealNumber, estimator.priors)))

@sym_predict.register(QuantileEstimator)
def sym_predict_quantile_estimator(estimator):
    return Function(syms(estimator), tuple(), (RealNumber(estimator.quantile),))

@sym_predict.register(LogOddsEstimator)
def sym_predict_log_odds_estimator(estimator):
    return Function(syms(estimator), tuple(), (RealNumber(estimator.prior),))

@sym_predict.register(MeanEstimator)
def sym_predict_mean_estimator(estimator):
    return Function(tuple(), tuple(), (RealNumber(estimator.mean),))

@sym_predict.register(ZeroEstimator)
def sym_predict_zero_estimator(estimator):
    return Function(tuple(), tuple(), (Zero(),))

@syms.register(LossFunction)
def syms_loss_function(loss):
    return (Symbol('x'),)

@sym_score_to_proba.register(BinomialDeviance)
def sym_score_to_proba_binomial_deviance(loss):
    inputs = syms(loss)
    calls = tuple()
    outputs = (1 - Expit(inputs[0]), Expit(inputs[0]))
    return Function(inputs, calls, outputs)

@sym_score_to_decision.register(ClassificationLossFunction)
def sym_score_to_decision_binomial_deviance(loss):
    score_to_proba = sym_score_to_proba(loss)
    Var = VariableFactory()
    compl_proba, proba = (Var(), Var())
    inputs = (compl_proba, proba)
    calls = tuple()
    outputs = (Piecewise((One(), proba > compl_proba), (Zero(), True)),)
    return Function(inputs, calls, outputs).compose(score_to_proba)

