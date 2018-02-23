from sklearn.base import ClassifierMixin
import re
from ..dispatching import fallback, call_method_or_dispatch
from .expression import RealVariable, Piecewise, RealNumber
from sklearn2code.sym.expression import true, Variable, Integer
from sklearn2code.sym.function import Function

sym_decision_function_doc = '''
Parameters
----------
estimator : A scikit-learn or other compatible fitted classifier.

Returns
-------
Function
    A Function object specifying the decision function for estimator.

Raises
------
NotFittedError
    When the estimator is not fitted.

NotImplementedError
    When the estimator's type is not supported.

'''
sym_decision_function = call_method_or_dispatch('sym_decision_function', docstring=sym_decision_function_doc)

sym_predict_proba_doc = '''
Parameters
----------
estimator : A scikit-learn or other compatible fitted classifier.

Returns
-------
Function
    A Function object specifying the predict_proba function for estimator.

Raises
------
NotFittedError
    When the estimator is not fitted.

NotImplementedError
    When the estimator's type is not supported.

'''
sym_predict_proba = call_method_or_dispatch('sym_predict_proba', docstring=sym_predict_proba_doc)


def input_size_from_coef(estimator):
    coef = estimator.coef_
    n_inputs = coef.shape[-1]
    return n_inputs

def input_size_from_n_features_(estimator):
    return estimator.n_features_

def input_size_from_n_features(estimator):
    return estimator.n_features

input_size_doc = '''
Parameters
----------
estimator : A scikit-learn or other compatible fitted estimator.

Returns
-------
int
    The number of columns needed for prediction by estimator.

Raises
------
NotFittedError
    When the estimator is not fitted.

NotImplementedError
    When the estimator's type is not supported.

'''
input_size = call_method_or_dispatch('input_size', docstring=input_size_doc)
input_size.register(object, fallback(input_size_from_n_features_, input_size_from_n_features, input_size_from_coef, 
                                     exception_type=AttributeError))

sym_predict_doc = '''
Parameters
----------
estimator : A scikit-learn or other compatible fitted classifier.

Returns
-------
Function
    A Function object specifying the predict function for estimator.

Raises
------
NotFittedError
    When the estimator is not fitted.

NotImplementedError
    When the estimator's type is not supported.

'''
sym_predict = call_method_or_dispatch('sym_predict', docstring=sym_predict_doc)
# sym_predict_proba(estimator).select_outputs(1).apply(lambda x: Piecewise((RealNumber(1), x >= RealNumber(.5)), (RealNumber(0), true)))

sym_score_to_decision_doc = '''
Parameters
----------
loss : A scikit-learn LossFunction or other compatible loss function.

Returns
-------
Function
    A Function object specifying the relationship between score and decision_function for estimator.

Raises
------
NotImplementedError
    When the loss's type is not supported.
'''
sym_score_to_decision = call_method_or_dispatch('sym_score_to_decision', docstring=sym_score_to_decision_doc)

sym_score_to_proba_doc = '''
Parameters
----------
loss : A scikit-learn LossFunction or other compatible loss function.

Returns
-------
Function
    A Function object specifying the relationship between score and predict_proba for estimator.

Raises
------
NotImplementedError
    When the loss's type is not supported.
'''
sym_score_to_proba = call_method_or_dispatch('sym_score_to_proba', docstring=sym_score_to_proba_doc)

sym_transform_doc = '''
Parameters
----------
estimator : A scikit-learn or other compatible fitted classifier.

Returns
-------
Function
    A Function object specifying the transform function for estimator.

Raises
------
NotFittedError
    When the estimator is not fitted.

NotImplementedError
    When the estimator's type is not supported.
'''
sym_transform = call_method_or_dispatch('sym_transform', docstring=sym_transform_doc)


sym_inverse_transform_doc = '''
estimator : A scikit-learn or other compatible fitted classifier.

Returns
-------
Function
    A Function object specifying the inverse_transform function for estimator.

Raises
------
NotFittedError
    When the estimator is not fitted.

NotImplementedError
    When the estimator's type is not supported.
'''
sym_inverse_transform = call_method_or_dispatch('sym_inverse_transform', docstring=sym_inverse_transform_doc)


syms_doc = '''
Parameters
----------
estimator : A scikit-learn or other compatible fitted estimator.

Returns
-------
tuple of Symbols
    The input symbols estimator.

Raises
------
NotFittedError
    When the estimator is not fitted.

NotImplementedError
    When the estimator's type is not supported.
'''
def syms_x(estimator):
    return tuple(RealVariable('x%d' % d) for d in range(input_size(estimator)))

def syms_xlabels(estimator):
    return tuple(map(RealVariable, estimator.xlabels_))

def syms_empty(estimator):
    return tuple()

syms = call_method_or_dispatch('syms', docstring=syms_doc)
syms.register(object, fallback(syms_xlabels, syms_x, syms_empty, 
                                    exception_type=(AttributeError, NotImplementedError)))



        
    
    

