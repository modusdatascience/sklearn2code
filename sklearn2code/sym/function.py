from toolz.functoolz import curry, flip as tzflip, identity, compose, complement
from toolz.curried import itemmap
from operator import methodcaller, __add__, __mul__, __sub__, __or__
from itertools import starmap
from six import PY2, PY3, string_types
from types import MethodType
from sklearn2code.dispatching import fallback
from .base import safe_symbol
from six.moves import reduce
from sklearn2code.sym.base import VariableFactory
from networkx.classes.digraph import DiGraph
import networkx
from toolz.dicttoolz import merge

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

def isiterable(obj):
    return hasattr(obj, '__iter__')

def tupify(obj):
    if isiterable(obj) and not isinstance(obj, string_types):
        return tuple(obj)
    else:
        return (obj,)

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
        self.inputs = tuple(map(safe_symbol, tupify(inputs)))
        self.calls = tupsmap(1, 
                             tupfun(identity, compose(tuple, curry(map)(safe_symbol))), 
                             tupsmap(0, compose(tuple, curry(map)(safe_symbol)), calls))
        self.outputs = tupify(outputs)
        self._validate()
    
    def map_symbols(self, symbol_map):
        new_inputs = self.map_input_symbols(symbol_map)
        new_calls = self.map_call_symbols(symbol_map)
        new_outputs = self.map_output_symbols(symbol_map)
        return Function(new_inputs, new_calls, new_outputs)
    
    def map_input_symbols(self, symbol_map):
        safe_sub_map = itemmap(tupfun(safe_symbol, safe_symbol), symbol_map)
        symbol_map_ = fallback(safe_sub_map.__getitem__, identity, exception_type=KeyError)
        return tuple(map(symbol_map_, self.inputs))
    
    def map_call_symbols(self, symbol_map):
        safe_sub_map = itemmap(tupfun(safe_symbol, safe_symbol), symbol_map)
        symbol_map_ = fallback(safe_sub_map.__getitem__, identity, exception_type=KeyError)
        symbol_tup_map = compose(tuple, curry(map)(symbol_map_))
        return tuple(map(tupfun(symbol_tup_map, tupfun(identity, symbol_tup_map)), self.calls))
    
    def map_output_symbols(self, symbol_map):
        safe_sub_map = itemmap(tupfun(safe_symbol, safe_symbol), symbol_map)
        return tuple(map(methodcaller('subs', safe_sub_map), self.outputs))
    
    def input_vars(self):
        return frozenset(self.inputs)
    
    def local_vars(self):
        return reduce(__or__, map(compose(frozenset, tupget(0)), self.calls), frozenset())
    
    def vars(self):
        return self.input_vars() | self.local_vars()
    
    def revar(self, existing):
        Var = VariableFactory(existing=existing)
        symbol_map = {var: Var() for var in self.vars()}
        return self.map_symbols(symbol_map)
    
    def trim(self, used=None):
        '''
        Remove unused computation.
        '''
        if used is None:
            used_ = frozenset(range(self.outputs))
        elif isinstance(used, string_types):
            used_ = frozenset((used,))
        else:
            used_ = frozenset(used)
        trimmed_outputs = tuple(map(self.outputs.__getitem__, sorted(used)))
        used_symbols = reduce(__or__, map(methodcaller('free_symbols'), (self.outputs[i] for i in used_)))
        trimmed_calls = tuple()
        for assigned, (fun, arguments) in reversed(self.calls):
            argmap = dict(zip(fun.inputs, arguments))
            trimmed_fun = fun.trim(used_symbols)
            trimmed_arguments = tuple(map(argmap.__getitem__, trimmed_fun.inputs))
            trimmed_assigned = filter(used_symbols.__contains__, assigned)
            trimmed_calls = (trimmed_assigned, (trimmed_fun, trimmed_arguments)) + trimmed_calls
            used_symbols = used_symbols | frozenset(trimmed_arguments)
        trimmed_inputs = tuple(filter(used_symbols.__contains__, self.inputs))
        return Function(trimmed_inputs, trimmed_calls, trimmed_outputs)
    
    def __eq__(self, other):
        if not isinstance(other, Function):
            return NotImplemented
        return self.inputs == other.inputs and set(self.calls) == set(other.calls) and self.outputs == other.outputs
    
    def __hash__(self):
        return hash((self.inputs, self.calls, self.outputs))
    
    def digraph(self):
        g = DiGraph()
        g.add_node(self)
        for _, (fun, _) in self.calls:
            g.add_node(fun)
            g.add_edge(fun, self)
            g = networkx.compose(g, fun.digraph())
        return g
    
    def compose(self, right):
        '''
        Compose.  Assume the outputs of right match, in order, the inputs of self.
        '''
        left = self.revar(right.vars())
        inputs = right.inputs
        calls = ((left.inputs, (right, right.inputs)),) + left.calls
        outputs = left.outputs
        return Function(inputs, calls, outputs)
    
    def select_outputs(self, selection):
        return Function(self.inputs, self.calls, tupify(self.outputs[selection]))
    
    def append_inputs(self, inputs):
        overlap = self.vars() & frozenset(inputs)
        if overlap:
            raise ValueError('Overlapping variables when appending inputs: %s' % str(overlap))
        return Function(self.inputs + inputs, self.calls, self.outputs)
    
    def _merge_calls(self, other):
        '''
        Assume non-overlapping local variables.
        '''
        self.ensure_same_inputs(other)
        other_calls = other.calls
        symbol_map = dict()
        result = self.calls
        existing_call_map = dict(map(reversed, self.calls))
        for assigned, (called, passed) in other_calls:
            if (called, passed) in existing_call_map:
                symbol_map = merge(symbol_map, dict(zip(assigned, existing_call_map[(called, passed)])))
            else:
                result += ((assigned, (called, passed)),)
        return result, symbol_map
        
#         return (self.calls + tuple(filter(complement(set(self.calls).__contains__), 
#                                          other.map_symbols(symbol_map).calls)),
#                 symbol_map)
    
    def ensure_same_inputs(self, other):
        if self.inputs != other.inputs:
                raise ValueError('Inputs don\'t match: %s != %s' % 
                                 (str(self.inputs), str(other.inputs)))
    
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
    
    def cartesian_product(self, other):
        self.ensure_same_inputs(other)
        inputs = self.inputs
        local_vars = self.local_vars()
        Var = VariableFactory(existing=(other.vars() | local_vars))
        other = other.map_symbols({var: Var() for var in local_vars})
        calls, symbol_map = self._merge_calls(other)
        outputs = self.outputs + other.map_output_symbols(symbol_map)
        return Function(inputs, calls, outputs)
    
def cart(*funs):
    return reduce(lambda x,y: x.cartesian_product(y), funs)

def funop(op, cls, name, flip=False):
    def __op__(self, other):
        if isinstance(other, Function):
            self.ensure_same_inputs(other)
            local_vars = self.local_vars()
            Var = VariableFactory(existing=(other.vars() | local_vars))
            other = other.map_symbols({var: Var() for var in local_vars})
            self.ensure_same_output_length(other)
            calls, symbol_map = self._merge_calls(other)
            outputs = tuple(starmap(op, zip(self.outputs, other.map_output_symbols(symbol_map))))
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
