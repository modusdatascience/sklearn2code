from abc import ABCMeta, abstractmethod
from six import with_metaclass
from sklearn2code.utility import tupify


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
    def __init__(self):
        if self.__class__ is Expression:
            raise NotImplementedError('Attempt to instantiate abstract class.')
    
    @property
    @abstractmethod
    def outtype(self):
        raise NotImplementedError()
    
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
    
    @property
    def e(self):
        return Equaler(self)
    
    @property
    def x(self):
        return self


class OutType(object):
    pass

class RealOut(OutType):
    pass

class IntOut(OutType):
    pass

class BoolOut(OutType):
    pass

class StrOut(OutType):
    pass

class EnumOutBase(OutType):
    pass

def enum_type(name, values):
    return type(name, (EnumOutBase,), {'values': tupify(values)})

class RealExpression(Expression):
    outtype = RealOut

class IntExpression(Expression):
    outtype = IntOut

class BoolExpression(Expression):
    outtype = BoolOut

class StrExpression(Expression):
    outtype = StrOut

# class EnumExpressionBase(Expression):
#     pass
# 
# def enum_expression_type(name, outtype):
#     return type(name, (EnumExpressionBase,), {'outtype': outtype})

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

class RealVariable(RealExpression, Variable):
    pass

class IntVariable(IntExpression, Variable):
    pass

class BoolVariable(IntExpression, Variable):
    pass

class StrVariable(IntExpression, Variable):
    pass

class EnumVariableBase(Variable):
    pass

def enum_variable_type(name, outtype):
    return type(name, (EnumVariableBase,), {'outtype': outtype})

class Constant(Expression):
    def __init__(self, value):
        self.value = value

class RealNumber(RealExpression, Constant):
    def __init__(self, value):
        if not float(value) == value:
            raise ValueError('Invalid value for constant.')
        super(RealNumber, self).__init__(float(value))
        
class Integer(IntExpression, Constant):
    def __init__(self, value):
        if not int(value) == value:
            raise ValueError('Invalid value for constant.')
        super(Integer, self).__init__(int(value))

class Boolean(BoolExpression, Constant):
    def __init__(self, value):
        if not bool(value) == value:
            raise ValueError('Invalid value for constant.')
        super(Boolean, self).__init__(bool(value))

class Str(StrExpression, Constant):
    def __init__(self, value):
        if not str(value) == value:
            raise ValueError('Invalid value for constant.')
        super(Str, self).__init__(str(value))

class EnumConstantBase(Constant):
    def __init__(self, value):
        if not (str(value) == value) and (str(value) in self.outtype.values):
            raise ValueError('Invalid value for constant.')
        super(Str, self).__init__(str(value))
        
    def validate(self, value):
        return (str(value) == value) and (str(value) in self.outtype.values)

def enum_constant_type(name, outtype):
    return type(name, (EnumConstantBase,), {'outtype': outtype})

class AryFunctionBase(Expression):
    pass

class ZeroaryFunction(AryFunctionBase):
    pass




    

