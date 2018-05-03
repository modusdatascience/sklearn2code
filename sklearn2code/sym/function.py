from toolz.functoolz import curry, flip as tzflip, identity, compose,\
    flip
from toolz.curried import itemmap
from operator import methodcaller, __add__, __mul__, __sub__, __or__,\
    __getitem__
from itertools import starmap
from six import PY2, PY3
from types import MethodType
from sklearn2code.dispatching import fallback
from six.moves import reduce
from networkx.classes.digraph import DiGraph
import networkx
from toolz.dicttoolz import merge
from sklearn2code.utility import tupify, tupsmap, tupfun, tupget
from sklearn2code.sym.expression import Variable, RealVariable
import re
from networkx.algorithms.operators.all import compose_all

def safe_symbol(s):
    if isinstance(s, Variable):
        return s
    return RealVariable(s)

class VariableFactory(object):
    def __init__(self, base='x', existing=[]):
        self.base = base
        self.existing = set(map(safe_symbol, existing))
        self.current_n = self._get_current_n()
    
    def _get_current_n(self):
        regex = re.compile('%s(\d+)' % self.base)
        result = -1
        for sym in self.existing:
            match = regex.match(sym.name)
            if match:
                val = int(match.group(1))
                if val > result:
                    result = val
        result += 1
        return result
    
    def __call__(self):
        result = self.base + str(self.current_n)
        self.current_n += 1
        return RealVariable(result)

class VariableNameFactory(VariableFactory):
    def __call__(self):
        result = self.base + str(self.current_n)
        self.current_n += 1
        return result

def _create_index(function, index, counter=None):
    if counter is None:
        counter = [0]
    index[function] = counter[0]
    counter[0] += 1
    for _, (fun, _) in function.calls:
        _create_index(fun, index, counter)

def toposort(functions):
    counter = [0]
    index = {}
    for function in functions:
        _create_index(function, index, counter)
    inverse_index = dict(list(map(reversed, index.items())))
    digraphs = list(map(methodcaller('_digraph', index), functions))
    digraph = compose_all(digraphs)
    return tuple(map(inverse_index.__getitem__, networkx.topological_sort(digraph)))

class Function(object):
    def __init__(self, inputs, calls, outputs, origin=None):
        '''
        A Function represents a function in the computational sense.  Function objects are the 
        intermediary between fitted estimators and generated code.  Adapters return Function 
        objects, and sklearn2code converts Function objects into working code.  A Function object
        is composed of Expression objects (including Variable objects) and other Function objects.  
        It knows its inputs (Variable objects), its internal calls (made up of Variable objects 
        and other Function objects), and its outputs (general Expression objects).  
        
        Parameters
        ----------
        inputs : tuple of Variables
            The input variables for this function.
        
        calls : tuple of pairs with (tuples of Variables, pairs of Function objects and tuples of their inputs)
            The values are other function calls made by this function.  The keys are 
            variables to which the outputs are assigned.  The number of output variables in the
            key must match the number of outputs in the Function.  The length of the tuple of inputs must match the
            number of inputs for the function.  Also, no two keys may contain 
            the same variable.  These constraints are checked.
            
        outputs : tuple of expressions
            The actual calculations made by this Function.  The return values of the function
            are the results of the computations expressed by the expressions.
        
        '''
        self.inputs = tuple(map(safe_symbol, tupify(inputs)))
        self.calls = tupsmap(1, 
                             tupfun(identity, compose(tuple, curry(map)(safe_symbol))), 
                             tupsmap(0, compose(tuple, curry(map)(safe_symbol)), calls))
        self.outputs = tupify(outputs)
        self._validate()
    
    def all_variables(self):
        result = set()
        result |= set(self.inputs)
        result |= reduce(__or__, map(compose(set, flip(__getitem__)(0)), self.calls), set())
        return result
    
    def __str__(self):
        return 'Function(%s, %s, %s)' % ('('+', '.join(map(str, self.inputs)) + ')', 
                                         '('+', '.join(map(str, self.calls)) + ')', 
                                         '('+', '.join(map(str, self.outputs)) + ')', 
                                         )
        
    def __repr__(self):
        return str(self)
#         return 'Function(%s, %s, %s)' % (repr(self.inputs), repr(self.calls), repr(self.outputs))
    
    @classmethod
    def from_expression(cls, expr):
        inputs = sorted(expr.free_symbols, key=flip(getattr)('name'))
        calls = tuple()
        outputs = (expr,)
        return Function(inputs, calls, outputs)
    
    @classmethod
    def from_expressions(cls, exprs):
        inputs = sorted(reduce(__or__, map(flip(getattr)('free_symbols'), exprs), frozenset()), 
                        key=flip(getattr)('name'))
        calls = tuple()
        outputs = tuple(exprs)
        return Function(inputs, calls, outputs)
    
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
            used_ = frozenset(range(len(self.outputs)))
        else:
            used_ = frozenset(used)
        trimmed_outputs = tuple(map(self.outputs.__getitem__, sorted(used_)))
        used_symbols = reduce(__or__, map(flip(getattr)('free_symbols'), trimmed_outputs), frozenset())
        trimmed_calls = tuple()
        for assigned, (fun, arguments) in reversed(self.calls):
            argmap = dict(zip(fun.inputs, arguments))
            trimmed_assigned = tuple(filter(used_symbols.__contains__, assigned))
            if not trimmed_assigned:
                continue
            trimmed_fun = fun.trim(frozenset(i for i in range(len(assigned)) if assigned[i] in used_symbols))
            trimmed_arguments = tuple(map(argmap.__getitem__, trimmed_fun.inputs))
            trimmed_calls = ((trimmed_assigned, (trimmed_fun, trimmed_arguments)),) + trimmed_calls
            used_symbols = used_symbols | frozenset(trimmed_arguments)
        trimmed_inputs = tuple(filter(used_symbols.__contains__, self.inputs))
        return Function(trimmed_inputs, trimmed_calls, trimmed_outputs)
    
    def __eq__(self, other):
        if not isinstance(other, Function):
            return NotImplemented
        return self.inputs == other.inputs and set(self.calls) == set(other.calls) and self.outputs == other.outputs
    
    def __hash__(self):
        return hash((self.inputs, self.calls, self.outputs))
    
    def _digraph(self, index):
        g = DiGraph()
        g.add_node(index[self])
        for _, (fun, _) in self.calls:
            g.add_node(index[fun])
            g.add_edge(index[fun], index[self])
            g = networkx.compose(g, fun._digraph(index))
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
                result += ((assigned, (called, 
                                       tuple(map(fallback(symbol_map.__getitem__, identity, exception_type=KeyError), 
                                                 passed)))),)
        return result, symbol_map
        
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

def comp(*funs):
    return reduce(lambda x,y: x.compose(y), funs)

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
