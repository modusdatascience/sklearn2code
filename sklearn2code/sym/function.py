from toolz.functoolz import curry, flip as tzflip, identity, compose, complement
from frozendict import frozendict
from toolz.curried import valmap, itemmap
from operator import methodcaller, __add__, __mul__, __sub__
from itertools import repeat, starmap
from toolz.dicttoolz import merge_with
from six import PY2, PY3
from types import MethodType
from sklearn2code.dispatching import fallback
from sympy.core.symbol import Symbol

def safe_symbol(s):
    if isinstance(s, Symbol):
        return s
    return Symbol(s)

@curry
def tupsmap(n, fun, tups):
    return tuple([tup[:n] + (fun(tup[n]),) + tup[(n+1):] for tup in tups])

@curry
def tupapply(tup):
    return tup[0](*tup[1:])

def tupfun(*funs):
    def _tupfun(tup):
        return tuple(map(tupapply, zip(funs, tup)))
    return _tupfun

@curry
def tupget(n, tup):
    return tup[n]
    
class Function(object):
    def __init__(self, inputs, calls, outputs, origin=None):
        '''
        Parameters
        ----------
        inputs : tuple of sympy Symbols
            The input variables for this function.
        
        calls : tuple of pairs with (tuples of Symbols, pairs of Function objects and tuples of their inputs)
            The values are other function calls made by this function.  The keys are 
            variables to which the outputs are assigned.  The number of output symbols in the
            key must match the number of outputs in the Function.  The length of the tuple of inputs must match the
            number of inputs for the function.  Also, no two keys may contain 
            the same variable symbol.  These constraints are checked.
            
        outputs : tuple of sympy expressions
            The actual calculations made by this Function.  The return values of the function
            are the results of the computations expressed by the expressions.
        
        '''
        self.inputs = tuple(map(safe_symbol, inputs))
        self.calls = tupsmap(1, 
                             tupfun(identity, compose(tuple, curry(map)(safe_symbol))), 
                             tupsmap(0, compose(tuple, curry(map)(safe_symbol)), calls))
        self.outputs = tuple(outputs)
#         self.origin = origin
        self._validate()
#         self.sym = SymFunctionInterface(self)
    
    def __eq__(self, other):
        if not isinstance(other, Function):
            return NotImplemented
        return self.inputs == other.inputs and set(self.calls) == set(other.calls) and self.outputs == other.outputs
    
    def __hash__(self):
        return hash((self.inputs, self.calls, self.outputs))
    
    def compose(self, right):
        '''
        Compose.  Assume the outputs of right match, in order, the inputs of left.
        '''
        inputs = right.inputs
        calls = ((self.inputs, (right, right.inputs)),) + self.calls
        outputs = self.outputs
        return Function(inputs, calls, outputs)
    
    def map_symbols(self, symbol_map):
        symbol_map_ = itemmap(tupfun(safe_symbol, safe_symbol), symbol_map)
        new_inputs = tuple(map(fallback(symbol_map_.__getitem__, identity, exception_type=KeyError), self.inputs))
        new_calls = tupsmap(1, tupfun(identity, curry(map)(fallback(symbol_map_.__getitem__, identity, exception_type=KeyError))), self.calls)
        new_outputs = tuple(map(methodcaller('subs', symbol_map_), self.outputs))
        return Function(new_inputs, new_calls, new_outputs)
        
    def concat_inputs(self, other):
        return Function(self.inputs + other.inputs, self.calls, self.outputs)
    
    def concat_calls(self, other):
        return Function(self.inputs, self.calls + other.calls, self.outputs)
    
    def _merge_calls(self, other):
        return self.calls + tuple(filter(complement(set(self.calls).__contains__), other.calls))
    
    def concat_outputs(self, other):
        return Function(self.inputs, self.calls, self.outputs + other.outputs)
    
    def ensure_same_inputs(self, other):
        if self.inputs != other.inputs:
                raise ValueError('Inputs don\'t match: %s != %s' % 
                                 (str(self.inputs, other.inputs)))
    
    def ensure_same_output_length(self, other):
        if len(self.outputs) != len(other.outputs):
                raise ValueError('Output lengths don\'t match: %d != %d' % 
                                 (len(self.outputs), len(other.outputs)))
    
    def ensure_no_call_collisions(self, other):
        overlap = set(map(tupget(0), self.calls)) & set(map(tupget(0), other.calls))
        if overlap:
            raise ValueError('Calls assign to overlapping symbolic variables: %s' % str(overlap))
    
    def _validate(self):
        sym_set = set(self.inputs)
        for syms, (function, inputs) in self.calls:
            for sym in syms:
                if sym in sym_set:
                    raise ValueError('Assigned to already used symbolic variable: %s' % str(sym))
                sym_set.add(sym)
            if len(syms) != len(function.outputs):
                raise ValueError('Output size of function does not match number of assigned variables: %d != %d' 
                                 % (len(syms), len(function.outputs)))
            if len(inputs) != len(function.inputs):
                raise ValueError('Input size of function does not match number of inputs: %d != %d'
                                 % (len(inputs, len(function.inputs))))
    
    def apply(self, fun):
        return Function(self.inputs, self.calls, tuple(map(fun, self.outputs)))
    
def funop(op, cls, name, flip=False):
    def __op__(self, other):
        if isinstance(other, Function):
            self.ensure_same_inputs(other)
            self.ensure_same_output_length(other)
            calls = self._merge_calls(other)
            outputs = tuple(starmap(op, zip(self.outputs, other.outputs)))
        else:
            calls = self.calls
            outputs = tuple(map((curry(tzflip(op)) if not flip else curry(op))(other), self.outputs))
        
        return Function(inputs=self.inputs, calls=calls, outputs=outputs)
    
    __op__.__name__ = name
    if PY2:
        __op__ = MethodType(__op__, None, cls)
    return __op__

Function.__add__ = funop(__add__, Function, '__add__')
Function.__radd__ = funop(__add__, Function, '__add__', flip=True)
Function.__mul__ = funop(__mul__, Function, '__mul__')
Function.__rmul__ = funop(__mul__, Function, '__mul__', flip=True)
Function.__sub__ = funop(__sub__, Function, '__sub__')
Function.__rsub__ = funop(__sub__, Function, '__sub__', flip=True)
if PY3:
    from operator import __truediv__  # @UnresolvedImport
    Function.__truediv__ = funop(__truediv__, Function, '__truediv__')
    Function.__rtruediv__ = funop(__truediv__, Function, '__truediv__', flip=True)
else:
    from operator import __div__  # @UnresolvedImport
    Function.__div__ = funop(__div__, Function, '__div__')
    Function.__rdiv__ = funop(__div__, Function, '__div__', flip=True)
