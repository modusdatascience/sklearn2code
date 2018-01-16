from abc import abstractmethod, ABCMeta
from ..dispatching import call_method_or_dispatch, create_registerer
from toolz.dicttoolz import merge_with
from itertools import starmap, repeat
from operator import __add__, __mul__, __sub__, methodcaller
from frozendict import frozendict
from six import PY3, PY2
from types import MethodType
from toolz.functoolz import curry, flip as tzflip
from toolz.curried import valmap


# def symfunop(op, cls, name, flip=False):
#     def __op__(self, other):
#         def eqq(x, y):
#             if x != y:
#                 raise ValueError()
#             return x
#         
#         if isinstance(other, SymFunctionParent):
#             if self.function.inputs != other.sym.function.inputs:
#                 raise ValueError()
#             if len(self.outputs) != len(other.sym.function.outputs):
#                 raise ValueError()
#             calls = frozendict(*merge_with(eqq, self.function.calls, other.sym.function.calls).items())
#             output = tuple(starmap(op, zip(self.function.outputs, other.sym.function.calls)))
#         else:
#             calls = self.function.calls
#             output = tuple(map((tzflip(curry(op)) if flip else curry(op))(other), self.function.output))
#         
#         return Function(inputs=self.function.inputs, calls=calls, output=output, 
#                         origin=(self.function.origin, other.sym.function.origin))
#     
#     __op__.__name__ = name
#     if PY2:
#         __op__ = MethodType(__op__, None, cls)
#     return __op__

# class SymFunctionParent(object):
#     pass

# class SymFunctionInterface(SymFunctionParent):
#     def __init__(self, function):
#         self.function = function
#         self.sym = self

# SymFunctionInterface.__add__ = symfunop(__add__, SymFunctionInterface, '__add__')
# SymFunctionInterface.__mul__ = symfunop(__mul__, SymFunctionInterface, '__mul__')
# SymFunctionInterface.__sub__ = symfunop(__sub__, SymFunctionInterface, '__sub__')
# if PY3:
#     from operator import __truediv__  # @UnresolvedImport
#     SymFunctionInterface.__truediv__ = symfunop(__truediv__, SymFunctionInterface, '__truediv__')
# else:
#     from operator import __div__  # @UnresolvedImport
#     SymFunctionInterface.__div__ = symfunop(__div__, SymFunctionInterface, '__div__')

# 
# def comp(left, right):
#     '''
#     Compose.  Assume the outputs of right match, in order, the inputs of left.
#     '''
#     inputs = right.inputs
#     calls = ((left.inputs, (right, frozendict(zip(right.inputs, right.inputs)))),) + left.calls
#     outputs = left.outputs
#     return Function(inputs, calls, outputs, origin=(left.origin, right.origin))
    
#     new_inputs = []
#     for symbol in left.inputs:
#         if symbol not in symbols:
#             new_inputs.append(symbols)
#     
    

class NamingSchemeBase(object):
    __metaclass__ = ABCMeta
    def name(self, function):
        '''
        Assign names to Function and any Functions called by Function.
        
        Parameters
        ----------
        function : instance of Function
        
        Returns
        -------
        dict with keys Functions and values strs
        '''

class SerializerBase(object):
    __metaclass__ = ABCMeta
    def serialize(self, functions):
        '''
        Serialize the function.
        
        Parameters
        ----------
        functions : dict with keys strs and values instances of Function.  The keys are typically used to 
        name the functions.
        
        Returns
        -------
        str
            A string representing the serialized Functions.  Usually this should be written to a file.
        '''
        return self._serialize(functions)
    
    @abstractmethod
    def _serialize(self, function):
        pass
    
class PrinterTemplateSerializer(SerializerBase):
    def __init__(self, printer, template):
        '''
        Parameters
        ----------
        printer : An instance of a CodePrinter subclass from sympy.
        template : A mako template.
        
        '''
        self.printer = printer
        self.template = template




sym_decision_function_dispatcher = {}
sym_decision_function = call_method_or_dispatch('sym_decision_function', sym_decision_function_dispatcher)
register_sym_decision_function = create_registerer(sym_decision_function_dispatcher, 'register_sym_decision_function')

sym_predict_proba_dispatcher = {}
sym_predict_proba = call_method_or_dispatch('sym_predict_proba', sym_predict_proba_dispatcher)
register_sym_predict_proba = create_registerer(sym_predict_proba_dispatcher, 'register_sym_predict_proba')


def input_size_from_coef(estimator):
    coef = estimator.coef_
    n_inputs = coef.shape[-1]
    return n_inputs

def input_size_from_n_features_(estimator):
    return estimator.n_features_

def input_size_from_n_features(estimator):
    return estimator.n_features

input_size_dispatcher = {}
input_size = call_method_or_dispatch('input_size', input_size_dispatcher)
register_input_size = create_registerer(input_size_dispatcher, 'register_input_size')

sym_predict_dispatcher = {}
sym_predict = call_method_or_dispatch('sym_predict', sym_predict_dispatcher)
register_sym_predict = create_registerer(sym_predict_dispatcher, 'register_sym_predict')

sym_score_to_decision_dispatcher = {}
sym_score_to_decision = call_method_or_dispatch('sym_score_to_decision', sym_score_to_decision_dispatcher)
register_sym_score_to_decision = create_registerer(sym_score_to_decision_dispatcher, 'register_sym_score_to_decision')

sym_score_to_proba_dispatcher = {}
sym_score_to_proba = call_method_or_dispatch('sym_score_to_proba', sym_score_to_proba_dispatcher)
register_sym_score_to_proba = create_registerer(sym_score_to_proba_dispatcher, 'register_sym_score_to_proba')

sym_transform_dispatcher = {}
sym_transform = call_method_or_dispatch('sym_transform', sym_transform_dispatcher)
register_sym_transform = create_registerer(sym_transform_dispatcher, 'register_sym_transform')






        
    
    

