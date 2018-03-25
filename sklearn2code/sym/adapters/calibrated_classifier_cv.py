from sklearn.calibration import CalibratedClassifierCV, _CalibratedClassifier
from sklearn2code.sym.base import sym_predict_proba, syms, sym_decision_function,\
    sym_predict
from sklearn.exceptions import NotFittedError
from operator import __add__
from sklearn2code.sym.expression import RealNumber, RealVariable
from six.moves import reduce
from sklearn2code.sym.function import Function, VariableFactory, comp

@sym_predict_proba.register(CalibratedClassifierCV)
def sym_predict_proba_calibrated_classifier_cv(estimator):
    if not hasattr(estimator, 'calibrated_classifiers_'):
        raise NotFittedError()
    return reduce(__add__, map(sym_predict_proba, estimator.calibrated_classifiers_)) / RealNumber(len(estimator.calibrated_classifiers_))

@syms.register(CalibratedClassifierCV)
def syms_calibrated_classifier_cv(estimator):
    return syms(estimator.calibrated_classifiers_[0])

@syms.register(_CalibratedClassifier)
def syms__calibrated_classifier(estimator):
    return syms(estimator.base_estimator)

@sym_predict_proba.register(_CalibratedClassifier)
def sym_predict_proba__calibrated_classifier(estimator):
    if hasattr(estimator.base_estimator, 'decision_function'):
        inner_pred = sym_decision_function(estimator.base_estimator)
    elif hasattr(estimator.base_estimator, 'predict_proba'):
        inner_pred = sym_predict_proba(estimator.base_estimator)
#     inner_pred = fallback(sym_decision_function, sym_predict_proba)(estimator.base_estimator)
    pre_result = reduce(__add__, map(sym_predict, estimator.calibrators_)).compose(inner_pred) / RealNumber(len(estimator.calibrators_))
    Var = VariableFactory()
    n_classes = len(estimator.classes_)
    if n_classes == 2:
        x = Var()
        normalize = Function((x,),
                             tuple(),
                             (RealNumber(1) - x, x))
    else:
        vars_ = tuple(Var() for _ in range(len(pre_result.outputs)))
        total = Function(vars_, 
                         tuple(), 
                         (reduce(__add__, vars_),))
        t = Var()
        normalize = Function(vars_,
                             (((t,),(total, vars_)),),
                             tuple(v / t for v in vars_))
    return comp(normalize, pre_result)

#     result = RealNumber(0)
#     for cal in estimator.calibrators_:
#         variables = syms(cal)
#         if len(variables) != 1:
#             raise ValueError()
#         var = variables[0]
#         result += sym_predict(cal).subs({var: inner_pred})
#     return result / RealNumber(len(estimator.calibrators_))



