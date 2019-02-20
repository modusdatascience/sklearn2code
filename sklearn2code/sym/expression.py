from toolz.functoolz import flip, compose, curry
from functools import singledispatch, partial
from sklearn2code.utility import tupfun
from operator import methodcaller, __or__
from multipledispatch.dispatcher import Dispatcher
from six.moves import reduce  # @UnresolvedImport
from six import with_metaclass
from abc import ABCMeta, abstractmethod
from frozendict import frozendict
from toolz.curried import itemmap
from itertools import chain

def undefined(*args):
    raise NotImplementedError()

def dispatch(name):
    result = singledispatch(undefined)
    result.__name__ = name
    return result

def get_common_type(types):
    common_type = None
    for t in types:
        if common_type is None:
            common_type = t
        while not issubclass(t, common_type):
            common_type = common_type.__mro__[1]
    return common_type
       
class Equaler(object):
    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        if not isinstance(other, Expression):
            return NotImplemented
        return Equals(self.x, other.x)
    
    @property
    def e(self):
        return self

class Expression(with_metaclass(ABCMeta, object)):
    '''
    Expression is an abstract base class for objects representing mathematical or 
    computational expressions (e.g., constants, variables, sums, logarithms).  The 
    Expression system was developed as a replacement for sympy expressions, which were
    used in a previous version, and shares some conventions with the sympy expression
    system.  For example, Expression objects implement the subs method for performing 
    variable substitution and have a free_symbols property which contains the Expression's
    free variables.
    
    Most operations on Expressions behave as you would expect.  For example, if `x` and `y` 
    are both NumberExpression then `x > y` is a BooleanExpression representing the comparison. 
    The exception to this is equality, which can't behave as expected because of compatibility 
    with Python's dictionary key system.  To get a BooleanExpression for equality, you can 
    write either `x.e == y` or `Equals(x, y)`.
    '''
    @abstractmethod
    def subs(self, varmap):
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def free_symbols(self):
        raise NotImplementedError()
    
    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError()
    
    @abstractmethod
    def __hash__(self):
        raise NotImplementedError()
    
    @abstractmethod
    def varfactory(self):
        raise NotImplementedError()
    
    @property
    def e(self):
        return Equaler(self)
    
    @property
    def x(self):
        return self

class UnaryFunction(Expression):
    def __init__(self, arg):
        super(UnaryFunction, self).__init__(arg)
        self.arg = arg
    
    def subs(self, varmap):
        return self.__class__(self.arg.subs(varmap))
    
    def __eq__(self, other):
        return self.__class__ is other.__class__ and self.arg == other.arg
    
    def __hash__(self):
        return hash((self.__class__, self.arg))
    
    @property
    def free_symbols(self):
        return self.arg.free_symbols
    
class BinaryFunction(Expression):
    def __init__(self, lhs, rhs):
        super(BinaryFunction, self).__init__(lhs, rhs)
        self.lhs = lhs
        self.rhs = rhs
    
    def subs(self, varmap):
        return self.__class__(self.lhs.subs(varmap), self.rhs.subs(varmap))
    
    def __eq__(self, other):
        return self.__class__ is other.__class__ and self.lhs == other.lhs and self.rhs == other.rhs
    
    def __hash__(self):
        return hash((self.__class__, self.lhs, self.rhs))
    
    @property
    def free_symbols(self):
        return self.lhs.free_symbols | self.rhs.free_symbols
    
class NaryFunction(Expression):
    def __init__(self, *args):
        super(NaryFunction, self).__init__(*args)
        self.args = tuple(args)
    
    def subs(self, varmap):
        return self.__class__(*map(methodcaller('subs', varmap), self.args))
    
    def __eq__(self, other):
        return self.__class__ is other.__class__ and self.args == other.args
    
    def __hash__(self):
        return hash((self.__class__, self.args))
    
    @property
    def free_symbols(self):
        return reduce(__or__, map(flip(getattr)('free_symbols'), self.args), set())
    
class Variable(Expression):
    def __init__(self, name):
        self.name = name
    
    @property
    def free_symbols(self):
        return set([self])
    
    def subs(self, varmap):
        return varmap.get(self, self)
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self.name)
    
    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.name == other.name
    
    def __hash__(self):
        return hash((self.__class__, self.name))
    
class PiecewiseBase(Expression):
    def __init__(self, *pairs):
        self.pairs = tuple(pairs)
        ExprType = self.outtype
        if not all(map(compose(all, tupfun(flip(isinstance)(ExprType), flip(isinstance)(BooleanExpression))), 
                              self.pairs)):
            raise TypeError('Arguments to Piecewise have incorrect type.')
    
    def subs(self, varmap):
        return self.__class__(*(map(tupfun(methodcaller('subs', varmap), methodcaller('subs', varmap)), self.pairs)))
    
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.__class___ == other.__class__ and self.pairs == other.pairs
    
    def __hash__(self):
        return hash((self.__class__, self.pairs))
    
    @property
    def free_symbols(self):
        return reduce(__or__, [expr.free_symbols | cond.free_symbols for expr, cond in self.pairs], set())
    
    def __str__(self):
        return (' else '.join(map(lambda pair: '(%s if %s' % 
                                  (str(pair[0]), str(pair[1])), self.pairs)) + 
                (')' * len(self.pairs)))

class BooleanExpression(Expression):
    def __and__(self, other):
        if not isinstance(other, BooleanExpression):
            return NotImplemented
        return And(self, other)
    
    def __or__(self, other):
        if not isinstance(other, BooleanExpression):
            return NotImplemented
        return Or(self, other)
    
    def __invert__(self, other):
        if not isinstance(other, BooleanExpression):
            return NotImplemented
        return Not(self, other)
    
    def varfactory(self):
        return BooleanVariable

class NumberExpression(Expression):
    def __gt__(self, other):
        if not isinstance(other, NumberExpression):
            return NotImplemented
        return Greater(self, other)
    
    def __ge__(self, other):
        if not isinstance(other, NumberExpression):
            return NotImplemented
        return GreaterEqual(self, other)
    
    def __lt__(self, other):
        if not isinstance(other, NumberExpression):
            return NotImplemented
        return Less(self, other)
    
    def __le__(self, other):
        if not isinstance(other, NumberExpression):
            return NotImplemented
        return LessEqual(self, other)
    
    def __add__(self, other):
        if not isinstance(other, NumberExpression):
            return NotImplemented
        return Sum(self, other)
    
    def __sub__(self, other):
        if not isinstance(other, NumberExpression):
            return NotImplemented
        return Difference(self, other)
    
    def __mul__(self, other):
        if not isinstance(other, NumberExpression):
            return NotImplemented
        return Product(self, other)
    
    def __truediv__(self, other):
        if not isinstance(other, NumberExpression):
            return NotImplemented
        return Quotient(self, other)
    
    def __neg__(self):
        return Negate(self)

class RealNumberExpression(NumberExpression):
    def varfactory(self):
        return RealVariable

class IntegerExpression(NumberExpression):
    def varfactory(self):
        return IntegerVariable

class StringExpression(Expression):
    def varfactory(self):
        return StringVariable

class RealPiecewise(RealNumberExpression, PiecewiseBase):
    outtype = RealNumberExpression

class BoolPiecewise(BooleanExpression, PiecewiseBase):
    outtype = BooleanExpression

class IntegerPiecewise(IntegerExpression, PiecewiseBase):
    outtype = IntegerExpression

def Piecewise(*args):
    if isinstance(args[0][0], RealNumberExpression):
        return RealPiecewise(*args)
    elif isinstance(args[0][0], BooleanExpression):
        return BoolPiecewise(*args)
    elif isinstance(args[0][0], IntegerExpression):
        return IntegerPiecewise(*args)
    elif isinstance(args[0][0], TupleExpression):
        return TuplePiecewise(*args)
    else:
        raise TypeError
 
class Constant(Expression):
    def __eq__(self, other):
        return self.__class__ == other.__class__
    
    def __hash__(self):
        return hash(self.__class__)
    
    @property
    def free_symbols(self):
        return set()
    
    def subs(self, varmap):
        return self
# 
# class VectorExpression(Expression):
#     class Component(Expression):
#         def __init__(self, vector, index):
#             self.vector = vector
#             self.index = index
#             if not isinstance(self.index, IntegerExpression):
#                 raise TypeError('Component index must be IntegerExpression.')
#             if not isinstance(self.vector, VectorExpression):
#                 raise TypeError('Component vector must be VectorExpression')
#     
#         def __str__(self):
#             return str(self.vector) + ('[%d]' % self.index)
#         
#         def __repr__(self):
#             return repr(self.vector) + ('[%d]' % self.index)
#         
#         def __eq__(self, other):
#             return type(self) is type(other) and self.index == other.index and self.vector == other.vector
#         
#         def __hash__(self):
#             return hash((self.vector, self.index))
#         
#         @property
#         def free_symbols(self):
#             return self.vector.free_symbols | self.index.free_symbols
#         
#         def subs(self, varmap):
#             return type(self)(self.vector.subs(varmap), self.index.subs(varmap))
#     
#     def __init__(self, dim):
#         self.dim = int(dim)
#         if self.dim != dim:
#             raise ValueError('dim must be an integer.')
#     
#     def __iter__(self):
#         for i in range(self.dim):
#             yield self[Integer(i)]
#     
#     @abstractmethod
#     def __getitem__(self, index):
#         pass
#     
#     def __add__(self, other):
#         if isinstance(other, NumberExpression):
#             return Vector(*[component + other for component in self])
#         if not isinstance(other, VectorExpression):
#             return NotImplemented
#         if self.dim != other.dim:
#             raise ValueError('Can\'t add VectorExpressions of different dimensions.')
#         return Vector(*[x + y for x, y in zip(self, other)])
#     
#     def __radd__(self, other):
#         if isinstance(other, NumberExpression):
#             return Vector(*[other + component for component in self])
#         if not isinstance(other, VectorExpression):
#             return NotImplemented
#         if self.dim != other.dim:
#             raise ValueError('Can\'t add VectorExpressions of different dimensions.')
#         return Vector(*[y + x for x, y in zip(self, other)])
#     
#     def __sub__(self, other):
#         if isinstance(other, NumberExpression):
#             return Vector(*[component - other for component in self])
#         if not isinstance(other, VectorExpression):
#             return NotImplemented
#         if self.dim != other.dim:
#             raise ValueError('Can\'t subtract VectorExpressions of different dimensions.')
#         return Vector(*[x - y for x, y in zip(self, other)])
#     
#     def __rsub__(self, other):
#         if isinstance(other, NumberExpression):
#             return Vector(*[other - component for component in self])
#         if not isinstance(other, VectorExpression):
#             return NotImplemented
#         if self.dim != other.dim:
#             raise ValueError('Can\'t subtract VectorExpressions of different dimensions.')
#         return Vector(*[y - x for x, y in zip(self, other)])
#     
#     def __mul__(self, other):
#         if isinstance(other, NumberExpression):
#             return Vector(*[component * other for component in self])
#         if not isinstance(other, VectorExpression):
#             return NotImplemented
#         if self.dim != other.dim:
#             raise ValueError('Can\'t multiply VectorExpressions of different dimensions.')
#         return Vector(*[x * y for x, y in zip(self, other)])
#     
#     def __rmul__(self, other):
#         if isinstance(other, NumberExpression):
#             return Vector(*[other * component for component in self])
#         if not isinstance(other, VectorExpression):
#             return NotImplemented
#         if self.dim != other.dim:
#             raise ValueError('Can\'t multiply VectorExpressions of different dimensions.')
#         return Vector(*[y * x for x, y in zip(self, other)])
#     
#     def __truediv__(self, other):
#         if isinstance(other, NumberExpression):
#             return Vector(*[__truediv__(component, other) for component in self])
#         if not isinstance(other, VectorExpression):
#             return NotImplemented
#         if self.dim != other.dim:
#             raise ValueError('Can\'t divide VectorExpressions of different dimensions.')
#         return Vector(*[__truediv__(x, y) for x, y in zip(self, other)])
#     
#     def __rtruediv__(self, other):
#         if isinstance(other, NumberExpression):
#             return Vector(*[__truediv__(other, component) for component in self])
#         if not isinstance(other, VectorExpression):
#             return NotImplemented
#         if self.dim != other.dim:
#             raise ValueError('Can\'t divide VectorExpressions of different dimensions.')
#         return Vector(*[__truediv__(y, x) for x, y in zip(self, other)])
#     
#     
#     def __and__(self, other):
#         if isinstance(other, NumberExpression):
#             return Vector(*[component & other for component in self])
#         if not isinstance(other, VectorExpression):
#             return NotImplemented
#         if self.dim != other.dim:
#             raise ValueError('Can\'t & VectorExpressions of different dimensions.')
#         return Vector(*[x & y for x, y in zip(self, other)])
#     
#     def __rand__(self, other):
#         if isinstance(other, NumberExpression):
#             return Vector(*[other & component for component in self])
#         if not isinstance(other, VectorExpression):
#             return NotImplemented
#         if self.dim != other.dim:
#             raise ValueError('Can\'t & VectorExpressions of different dimensions.')
#         return Vector(*[y & x for x, y in zip(self, other)])
#     
#     def __or__(self, other):
#         if isinstance(other, NumberExpression):
#             return Vector(*[component | other for component in self])
#         if not isinstance(other, VectorExpression):
#             return NotImplemented
#         if self.dim != other.dim:
#             raise ValueError('Can\'t | VectorExpressions of different dimensions.')
#         return Vector(*[x | y for x, y in zip(self, other)])
#     
#     def __ror__(self, other):
#         if isinstance(other, NumberExpression):
#             return Vector(*[other | component for component in self])
#         if not isinstance(other, VectorExpression):
#             return NotImplemented
#         if self.dim != other.dim:
#             raise ValueError('Can\'t | VectorExpressions of different dimensions.')
#         return Vector(*[y | x for x, y in zip(self, other)])
#     
#     def __invert__(self):
#         return Vector(*map(__invert__, self))
#     
#     def __neg__(self):
#         return Vector(*map(__neg__, self))



class FunctionOfType(Expression):
    def __init__(self, *args):
        if not all(map(flip(isinstance)(self.argtype), args)):
            raise TypeError('Attempt to create %s with arguments of incorrect output type.  Should be %s. Got %s.'
                            % (self.__class__.__name__, self.argtype.__name__, str(tuple(map(lambda x: x.__class__.__name__, args)))))

class FunctionOfBools(FunctionOfType):
    argtype = BooleanExpression

class FunctionOfReals(FunctionOfType):
    argtype = RealNumberExpression
    
class FunctionOfInts(FunctionOfType):
    argtype = IntegerExpression
    
class FunctionOfNumber(FunctionOfType):
    argtype = NumberExpression


# 
# class VectorNary(NaryFunction, VectorExpression):
#     '''
#     A vector-valued n-ary function.
#     '''
#     def __init__(self, *args):
#         NaryFunction.__init__(self, *args)
#         dim = len(self.args)
#         VectorExpression.__init__(self, dim)
# #     @property
# #     def dim(self):
# #         return Integer(len(self.args))
#     
#     def __eq__(self, other):
#         return self.__class__ is other.__class__ and all(starmap(__eq__, zip(self.args, other.args)))
#     
#     def __hash__(self):
#         return hash((self.__class__,) + tuple(map(hash, self.args)))
#     
#     @property
#     def free_symbols(self):
#         return reduce(__or__, map(flip(getattr)('free_symbols'), self.args), set())
#     
#     def subs(self, varmap):
#         return self.__class__(*map(methodcaller('subs', varmap), self.args))
#     
#     def sum(self):
#         return Sum(*(self.args))
#     
#     def product(self):
#         return Product(*(self.args))
# 
# class VectorExpressionReal(VectorExpression):
#     class Component(RealNumberExpression, VectorExpression.Component):
#         pass
# 
# class VectorExpressionInt(VectorExpression):
#     class Component(IntegerExpression, VectorExpression.Component):
#         pass
#     
# class VectorExpressionBool(VectorExpression):
#     class Component(BooleanExpression, VectorExpression.Component):
#         pass
# 
# 
# class FunctionOfVectors(FunctionOfType):
#     argtype = VectorExpression
# 
# class FunctionOfRealVectors(FunctionOfType):
#     argtype = VectorExpressionReal
#     
# class FunctionOfIntVectors(FunctionOfType):
#     argtype = VectorExpressionInt
# 
# class FunctionOfBoolVectors(FunctionOfType):
#     argtype = VectorExpressionBool

# class Dim(IntegerExpression, UnaryFunction, FunctionOfVectors):
#     pass
# 
# # TODO: Clean up the types
# class VectorSum(UnaryFunction, FunctionOfVectors):
#     pass
# 
# 
# class VectorExpressionOfReal(VectorExpressionReal, VectorNary, FunctionOfReals):
#     class Component(VectorExpressionReal.Component):
#         pass
# 
# class VectorExpressionOfInt(VectorExpressionInt, VectorNary, FunctionOfInts):
#     class Component(VectorExpressionInt.Component):
#         pass
#     
# class VectorExpressionOfBool(VectorExpressionBool, VectorNary, FunctionOfBools):
#     class Component(VectorExpressionBool.Component):
#         pass
# 
# class VectorBase(VectorExpression):
#     class Component(VectorExpression.Component):
#         pass
#     
#     def __getitem__(self, index):
#         if not isinstance(index, IntegerExpression):
#             raise TypeError('Vector index must be IntegerExpression.')
#         if isinstance(index, Integer):
#             return self.args[index.value]
#         return self.__class__.Component(self, index)
#     
#     def __str__(self):
#         return 'Vector(%s)' % (', '.join(map(str, self.args)))
#     
#     def __repr__(self):
#         return 'Vector(%s)' % (', '.join(map(repr, self.args)))
# 
# class VectorReal(VectorBase, VectorExpressionOfReal):
#     class Component(VectorExpressionReal.Component):
#         pass
# 
# class VectorInt(VectorBase, VectorExpressionOfInt):
#     class Component(VectorExpressionInt.Component):
#         pass
#     
# class VectorBool(VectorBase, VectorExpressionOfBool):
#     class Component(VectorExpressionBool.Component):
#         pass
# 
# Vector = dispatch('Vector')
# Vector.register(RealNumberExpression)(VectorReal)
# Vector.register(IntegerExpression)(VectorInt)
# Vector.register(BooleanExpression)(VectorBool)
# 
# class OrderedBase(VectorNary):
#     class Component(VectorExpression.Component):
#         pass
#     
#     
#     
#     def __str__(self):
#         return 'Ordered(%s)' % (', '.join(map(str, self.args)))
#     
#     def __repr__(self):
#         return 'Ordered(%s)' % (', '.join(map(repr, self.args)))
#     
#     def __getitem__(self, index):
#         return self.__class__.Component(self, index)
# 
# class OrderedReal(OrderedBase, VectorExpressionOfReal):
#     class Component(VectorExpressionOfReal.Component):
#         pass
# 
# class OrderedInt(OrderedBase, VectorExpressionOfInt):
#     class Component(VectorExpressionOfInt.Component):
#         pass
# 
# Ordered = dispatch('Ordered')
# Ordered.register(RealNumberExpression)(OrderedReal)
# Ordered.register(IntegerExpression)(OrderedInt)
#     
# class VectorVariable(VectorExpression, Variable):
#     def __getitem__(self, index):
#         return self.__class__.Component(self, index)
#     
#     def __init__(self, name, dim):
#         VectorExpression.__init__(self, dim)
#         Variable.__init__(self, name)
#         
# class RealVectorVariable(VectorVariable, VectorExpressionReal):
#     class Component(VectorExpressionReal.Component):
#         pass
# 
# class IntVectorVariable(VectorVariable, VectorExpressionInt):
#     class Component(VectorExpressionInt.Component):
#         pass
# 
# class BoolVectorVariable(VectorVariable, VectorExpressionBool):
#     class Component(VectorExpressionBool.Component):
#         pass

# class ScalarVariable(NumberExpression, Variable):
#     pass

class RealVariable(RealNumberExpression, Variable):
    pass

class Value(Constant):
    def __str__(self):
        return repr(self.value)
    
    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.value == other.value
    
    def __hash__(self):
        return hash((self.__class__, self.value))
    
class RealNumber(RealNumberExpression, Value):
    def __init__(self, value):
        if not float(value) == value:
            raise TypeError('Real value must be float.')
        self.value = float(value)

class Nan(RealNumberExpression, Constant):
    def __str__(self):
        return 'Nan'

nan = Nan()

class IsNan(BooleanExpression, UnaryFunction, FunctionOfReals):
    def __str__(self):
        return 'IsNan(%s)' % str(self.arg)

class StringVariable(StringExpression, Variable):
    pass

class String(StringExpression, Value):
    def __init__(self, value):
        if not str(value) == value:
            raise TypeError('String value must be str.')
        self.value = str(value)

class IntegerVariable(IntegerExpression, Variable):
    pass

class Integer(IntegerExpression, Value):
    def __init__(self, value):
        if not int(value) == value:
            raise TypeError('Integer value must be integer.')
        self.value = int(value)
    
class NegateBase(UnaryFunction):
    def __str__(self):
        return '-%s' % (self.arg)
    
class NegateReal(RealNumberExpression, NegateBase, FunctionOfReals):
    pass

class NegateInt(IntegerExpression, NegateBase, FunctionOfInts):
    pass

Negate = dispatch('Negate')
Negate.register(RealNumberExpression)(NegateReal)
Negate.register(IntegerExpression)(NegateInt)

class Log(RealNumberExpression, UnaryFunction, FunctionOfNumber):
    def __str__(self):
        return 'Log(%s)' % (self.arg)

class Expit(RealNumberExpression, UnaryFunction, FunctionOfNumber):
    def __str__(self):
        return 'Expit(%s)' % (self.arg)

# class OrderBase(NaryFunction):
#     def __init__(self, n, *args):
#         self.n = n
#         super(OrderBase, self).__init__(*args)
#     
#     def __str__(self):
#         return 'Order(%d)(%s)' % (self.n, ', '.join(map(str, self.args)))
#     
#     def subs(self, varmap):
#         return self.__class__(self.n.subs(varmap), *map(methodcaller('subs', varmap), self.args))
#     
#     def __eq__(self, other):
#         return super(OrderBase, self).__eq__(other) and self.n == other.n
#     
#     def __hash__(self):
#         return super(OrderBase, self).__hash__(self) + hash(self.n)
#     
#     @property
#     def free_symbols(self):
#         return super(OrderBase, self).free_symbols | set([self.n])
#     
# class OrderReal(RealNumberExpression, OrderBase, FunctionOfReals):
#     pass
# 
# class OrderInt(IntegerExpression, OrderBase, FunctionOfInts):
#     pass
# 
# Order = dispatch('Order')
# Order.register(IntegerExpression, RealNumberExpression)(OrderReal)
# Order.register(IntegerExpression, IntegerExpression)(OrderInt)

class MaxBase(NaryFunction):
    def __str__(self):
        return 'Max(%s)' % ', '.join(map(str, self.args))

class MaxReal(RealNumberExpression, MaxBase, FunctionOfReals):
    pass
    
class MaxInt(IntegerExpression, MaxBase, FunctionOfInts):
    pass

Max = dispatch('Max')
Max.register(RealNumberExpression)(MaxReal)
Max.register(IntegerExpression)(MaxInt)

class MinBase(NaryFunction):
    def __str__(self):
        return 'Min(%s)' % ', '.join(map(str, self.args))

class MinReal(RealNumberExpression, MinBase, FunctionOfReals):
    pass

class MinInt(IntegerExpression, MinBase, FunctionOfInts):
    pass

Min = dispatch('Min')
Min.register(RealNumberExpression)(MinReal)
Min.register(IntegerExpression)(MinInt)

class SumBase(NumberExpression, NaryFunction):
    def __str__(self):
        return '(%s)' % ' + '.join(map(str, self.args))
    
    def __add__(self, other):
        if not isinstance(other, Expression):
            return NotImplemented
        if isinstance(other, self.__class__):
            return self.__class__(*chain(self.args, other.args))
        else:
            return self.__class__(*self.args, other)

class SumReal(RealNumberExpression, SumBase, FunctionOfReals):
    pass

class SumInt(IntegerExpression, SumBase, FunctionOfInts):
    pass

Sum = dispatch('Sum')
Sum.register(RealNumberExpression)(SumReal)
Sum.register(IntegerExpression)(SumInt)
# Sum.register(VectorExpression)(methodcaller('sum'))

class DifferenceBase(NumberExpression, BinaryFunction):
    def __str__(self):
        return '(%s - %s)' % (self.lhs, self.rhs)
    
class DifferenceReal(RealNumberExpression, DifferenceBase, FunctionOfReals):
    pass

class DifferenceInt(IntegerExpression, DifferenceBase, FunctionOfInts):
    pass

Difference = dispatch('Difference')
Difference.register(RealNumberExpression)(DifferenceReal)
Difference.register(IntegerExpression)(DifferenceInt)

class ProductBase(NumberExpression, NaryFunction):
    def __str__(self):
        return '(%s)' % ' * '.join(map(str, self.args))
    
    def __mul__(self, other):
        if not isinstance(other, Expression):
            return NotImplemented
        if isinstance(other, self.__class__):
            return self.__class__(*chain(self.args, other.args))
        else:
            return self.__class__(*self.args, other)
    
class ProductReal(RealNumberExpression, ProductBase, FunctionOfReals):
    pass

class ProductInt(IntegerExpression, ProductBase, FunctionOfInts):
    pass

Product = dispatch('Product')
Product.register(RealNumberExpression)(ProductReal)
Product.register(IntegerExpression)(ProductInt)
# Product.register(VectorExpression)(methodcaller('product'))

class PowerBase(NumberExpression, BinaryFunction):
    def __str__(self):
        return '(%s ** %s)' % (self.lhs, self.rhs)

class IntPowerInt(IntegerExpression, PowerBase, FunctionOfInts):
    pass

class RealPowerInt(RealNumberExpression, PowerBase, FunctionOfInts):
    pass

class RealPowerReal(RealNumberExpression, PowerBase, FunctionOfReals):
    pass

Power = Dispatcher('Power')
Power.register(IntegerExpression, IntegerExpression)(IntPowerInt)
Power.register(RealNumberExpression, IntegerExpression)(RealPowerInt)
Power.register(RealNumberExpression, RealNumberExpression)(RealPowerReal)

class QuotientBase(NumberExpression, BinaryFunction):
    def __str__(self):
        return '(%s / %s)' % (self.lhs, self.rhs)

class QuotientReal(RealNumberExpression, QuotientBase, FunctionOfReals):
    pass

class QuotientInt(RealNumberExpression, QuotientBase, FunctionOfInts):
    pass

Quotient = dispatch('Quotient')
Quotient.register(RealNumberExpression)(QuotientReal)
Quotient.register(IntegerExpression)(QuotientInt)

class Boolean(BooleanExpression, Value):
    def __init__(self, value):
        self.value = bool(value)
        
    def __str__(self):
        return str(self.value)

true = Boolean(True)
false = Boolean(False)

class BooleanVariable(BooleanExpression, Variable):
    pass

class BoolToReal(RealNumberExpression, UnaryFunction, FunctionOfBools):
    def __str__(self):
        return 'BoolToReal(%s)' % self.arg

class And(BooleanExpression, NaryFunction, FunctionOfBools):
    def __str__(self):
        return '(%s)' % ' & '.join(map(str, self.args))

class Or(BooleanExpression, NaryFunction, FunctionOfBools):
    def __str__(self):
        return '(%s)' % ' | '.join(map(str, self.args))

class Not(BooleanExpression, UnaryFunction, FunctionOfBools):
    def __str__(self):
        return '(~%s)' % (self.arg)

class EqualsBase(BooleanExpression, BinaryFunction):
    def __str__(self):
        return '(%s == %s)' % (str(self.lhs), str(self.rhs))

class EqualsReal(FunctionOfReals):
    pass

class EqualsInt(FunctionOfInts):
    pass

class EqualsBool(FunctionOfBools):
    pass

Equals = dispatch('Equals')
Equals.register(RealNumberExpression)(EqualsReal)
Equals.register(IntegerExpression)(EqualsInt)
Equals.register(BooleanExpression)(EqualsBool)

class GreaterBase(BooleanExpression, BinaryFunction):
    def __str__(self):
        return '(%s > %s)' % (str(self.lhs), str(self.rhs))
    
class GreaterReal(GreaterBase, FunctionOfReals):
    pass

class GreaterInt(GreaterBase, FunctionOfReals):
    pass

class GreaterBool(GreaterBase, FunctionOfReals):
    pass

Greater = dispatch('Greater')
Greater.register(RealNumberExpression)(GreaterReal)
Greater.register(IntegerExpression)(GreaterInt)
Greater.register(BooleanExpression)(GreaterBool)

class GreaterEqualBase(BooleanExpression, BinaryFunction):
    def __str__(self):
        return '(%s >= %s)' % (str(self.lhs), str(self.rhs))
    
class GreaterEqualReal(GreaterEqualBase, FunctionOfReals):
    pass

class GreaterEqualInt(GreaterEqualBase, FunctionOfInts):
    pass

class GreaterEqualBool(GreaterEqualBase, FunctionOfBools):
    pass

GreaterEqual = dispatch('GreaterEqual')
GreaterEqual.register(RealNumberExpression)(GreaterEqualReal)
GreaterEqual.register(IntegerExpression)(GreaterEqualInt)
GreaterEqual.register(BooleanExpression)(GreaterEqualBool)

class LessEqualBase(BooleanExpression, BinaryFunction):
    def __str__(self):
        return '(%s <= %s)' % (str(self.lhs), str(self.rhs))
    
class LessEqualReal(LessEqualBase, FunctionOfReals):
    pass

class LessEqualInt(LessEqualBase, FunctionOfInts):
    pass

class LessEqualBool(LessEqualBase, FunctionOfBools):
    pass

LessEqual = dispatch('LessEqual')
LessEqual.register(RealNumberExpression)(LessEqualReal)
LessEqual.register(IntegerExpression)(LessEqualInt)
LessEqual.register(BooleanExpression)(LessEqualBool)

class LessBase(BooleanExpression, BinaryFunction):
    def __str__(self):
        return '(%s < %s)' % (str(self.lhs), str(self.rhs))

class LessReal(LessBase, FunctionOfReals):
    pass

class LessInt(LessBase, FunctionOfInts):
    pass

class LessBool(LessBase, FunctionOfBools):
    pass

Less = dispatch('Less')
Less.register(RealNumberExpression)(LessReal)
Less.register(IntegerExpression)(LessInt)
Less.register(BooleanExpression)(LessBool)

class FiniteMap(UnaryFunction):
    @curry
    def __init__(self, mapping, arg):
        self.mapping = frozendict(mapping)
        if not all(map(flip(isinstance)(Constant), chain(mapping.keys(), mapping.values()))):
            raise TypeError('Keys and values of FiniteMap must be Constants. Got %s.' % str(tuple(map(type, chain(mapping.keys(), mapping.values())))))
        self.arg = arg
        self.outtype = get_common_type(map(type, self.mapping.values()))
    
    @property
    def free_symbols(self):
        return reduce(__or__,
                      map(compose(curry(reduce)(__or__), 
                                    tupfun(flip(getattr)('free_symbols'), flip(getattr)('free_symbols'))), 
                            self.mapping.items())) | self.arg.free_symbols
    
    def subs(self, varmap):
        return self.__class__(mapping = itemmap(tupfun(methodcaller('subs', varmap=varmap), 
                                                       methodcaller('subs', varmap=varmap)), 
                                                self.mapping),
                              arg = self.arg.subs(varmap))
    
    def str(self):
        return 'Map(data={%s}, arg=%s)' % (', '.join(map(lambda x: str(x[0]) + ': ' + str(x[0]), 
                                                         self.mapping.items)), self.arg)

class Statistic(Expression):
    def __init__(self, data):
        self.data = tuple(data)
        self.outtype = get_common_type(map(type, data))
    
    def varfactory(self):
        return self.outtype.varfactory()
    
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.__class__ is other.__class__ and self.data == other.data
    
    def __hash__(self):
        return hash((self.__class__, self.data))
    
    @property
    def free_symbols(self):
        return reduce(__or__, map(flip(getattr('free_symbols')), self.data), set())
        
    def subs(self, varmap):
        return self.__class__(tuple(map(methodcaller('subs', varmap=varmap), self.data)))
    
    def str(self):
        return '%s(data=(%s,))' % (self.__class__.__name__, 
                                                  ', '.join(map(str, self.data)))

# class StatisticOfReals(Statistic):
#     def __init__(self, data):
#         if not all(map(flip(isinstance)(RealNumberExpression), data)):
#             raise TypeError('Elements of data should be of type RealNumberExpression. Got (%s)' % str(tuple(map(lambda x: x.__class__.__name__, data))))
#         super(StatisticOfReals, self).__init__(data)
        
        
class WeightedStatistic(Statistic):
    def __init__(self, data, weights):
        super(WeightedStatistic, self).__init__(data)
        self.weights = tuple(weights)
    
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.__class__ is other.__class__ and self.data == other.data and self.weights == other.weights
    
    def __hash__(self):
        return hash((self.__class__, self.data, self.weights))
    
    @property
    def free_symbols(self):
        return reduce(__or__, map(flip(getattr('free_symbols')), self.weights), set()) | super(WeightedStatistic, self).free_symbols
    
    def subs(self, varmap):
        return self.__class__(
                              data = tuple(map(methodcaller('subs', varmap=varmap), self.data)),
                              weights = tuple(map(methodcaller('subs', varmap=varmap), self.weights)),
                              )
        
    def str(self):
        return '%s(data=(%s,), weights=(%s,))' % (self.__class__.__name__, 
                                                  ', '.join(map(str, self.data)),
                                                  ', '.join(map(str, self.weights)))
        
class WeightedMode(WeightedStatistic):
    pass

class WeightedMedian(WeightedStatistic):
    pass

class TupleExpression(Expression):
    '''
    The dim of a tuple must be known at compile time.
    '''
    def __init__(self, dim):
        self.dim = int(dim)
        if self.dim != dim:
            raise ValueError('The dim must be an integer.')
    
    def __add__(self, other):
        if not isinstance(other, TupleExpression):
            return NotImplemented
        if isinstance(other, TupleSum):
            return other.__radd__(self)
        return TupleSum(self, other)
    
    def __radd__(self, other):
        if not isinstance(other, TupleExpression):
            return NotImplemented
        return TupleSum(self, other)
    
    def __mul__(self, other):
        if isinstance(other, NumberExpression):
            return ScalarMultiply(self, other)
        else:
            return NotImplemented
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, NumberExpression):
            return ScalarMultiply(self, RealNumber(1) / other)
        else:
            return NotImplemented
    
    def varfactory(self):
        return partial(TupleVariable, dim=self.dim)
    
class Tuple(TupleExpression, NaryFunction, FunctionOfNumber):
    def __init__(self, *args):
        TupleExpression.__init__(self, len(args))
        NaryFunction.__init__(self, *args)

class FunctionOfTuples(FunctionOfType):
    argtype = TupleExpression

class Total(UnaryFunction, FunctionOfTuples):
    pass

class NaryTupleFunction(TupleExpression, NaryFunction, FunctionOfTuples):
    def __init__(self, *args):
        NaryFunction.__init__(self, *args)
        if len(set(map(flip(getattr)('dim'), self.args))) > 1:
            raise ValueError('All arguments must have the same dim.')
        
class TupleSum(NaryTupleFunction):
    def __add__(self, other):
        if not isinstance(other, TupleExpression):
            return NotImplemented
        if isinstance(other, TupleSum):
            return TupleSum(*(self.args + other.args))
        else:
            return TupleSum(*(self.args + (other,)))
    
    def __radd__(self, other):
        if not isinstance(other, TupleExpression):
            return NotImplemented
        if isinstance(other, TupleSum):
            return TupleSum(*(other.args + self.args))
        else:
            return TupleSum(*((other,) + self.args))

class ScalarTupleOp(TupleExpression):
    def __init__(self, tup, scalar):
        self.tup = tup
        self.scalar = scalar
        if not isinstance(self.tup, TupleExpression):
            raise TypeError('tup must be a TupleExpression.')
        if not isinstance(self.scalar, NumberExpression):
            raise TypeError('scalar must be a NumberExpression.')
    
    def subs(self, varmap):
        self.tup.subs(varmap)
        self.scalar.subs(varmap)
    
    @property
    def free_symbols(self):
        return self.tup.free_symbols | self.scalar.free_symbols
    
    def __eq__(self, other):
        if not isinstance(self, type(other)):
            return NotImplemented
        return (type(self) is type(other) and 
                self.scalar == other.scalar and 
                self.tup == other.tup)
    
    def __hash__(self):
        return hash((type(self), self.scalar, self.tup))
    

class ScalarMultiply(ScalarTupleOp):
    pass

class TupleVariable(TupleExpression, Variable):
    def __init__(self, name, dim):
        TupleExpression.__init__(self, dim)
        Variable.__init__(self, name)

class TupleToTupleFunction(TupleExpression, UnaryFunction, FunctionOfTuples):
    def __init__(self, arg):
        TupleExpression.__init__(self, arg.dim)
        UnaryFunction.__init__(self, arg)

class Ordered(TupleToTupleFunction):
    pass

class NormalizeBase(TupleToTupleFunction):
    pass

class NormalizeL1(NormalizeBase):
    pass

class NormalizeL2(NormalizeBase):
    pass

class NormalizeLInf(NormalizeBase):
    pass

class TuplePiecewise(TupleExpression, PiecewiseBase):
    outtype = TupleExpression

# class TupleVariable(TupleBase):
#     pass
# 
# class Tuple(TupleBase):
#     def __iter__(self):



def as_value(obj):
    if isinstance(obj, str):
        return String(obj)
    elif isinstance(obj, float):
        return RealNumber(obj)
    elif isinstance(obj, bool):
        return Boolean(obj)
    elif isinstance(obj, int):
        return Integer(obj)
    else:
        raise ValueError('Conversion failed for object %s of type %s.' % (repr(obj), type(obj).__name__))

