from sklearn2code.sym.expression import RealNumber, Log,\
    PiecewiseBase, NegateBase, MaxBase,\
    MinBase, GreaterBase, GreaterEqualBase, LessEqualBase, LessBase, Nan, IsNan,\
    ProductBase, SumBase, QuotientBase, DifferenceBase, Value, Expit, And, Or,\
    Variable, BoolToReal, Not, FiniteMap, WeightedMode, RealPiecewise,\
    IntegerPiecewise, BoolPiecewise
from six import with_metaclass
from toolz.functoolz import curry
from multipledispatch.dispatcher import Dispatcher

@curry
def reduction(function_name, self, args):
    if len(args) == 1:
        return self(args[0])
    else:
        return function_name + '(' + self(args[0]) + ',' + reduction(function_name, self, args[1:]) + ')'

class InnerDispatcher(object):
    def __init__(self, parent, instance):
        self.parent = parent
        self.instance = instance
        
    def __call__(self, *args):
        return self.parent.dispatcher(self.instance, *args)
    
    def register(self, *types):
        return self.parent.register(*types)

class HeritableDispatcher(object):
    def __init__(self, name):
        self.dispatcher = Dispatcher(name)
        
    def register(self, *types):
        def _register(fun):
            fun._dispatcher_1f0irrij9 = self
            fun._dispatch_types_2m20fi4rin = tuple(types)
            return fun
        return _register
    
    def __get__(self, instance, owner):
        return InnerDispatcher(self, instance)

class ExpressionPrinterMeta(type):
    def __init__(self, name, bases, dct):
        for method in dct.values():
            if hasattr(method, '_dispatch_types_2m20fi4rin'):
                method._dispatcher_1f0irrij9.dispatcher.register(self, *method._dispatch_types_2m20fi4rin)(method)
        return type.__init__(self, name, bases, dct)

class ExpressionPrinter(with_metaclass(ExpressionPrinterMeta, object)):
    __call__ = HeritableDispatcher('__call__')

class BasicOperatorPrinter(ExpressionPrinter):
    @ExpressionPrinter.__call__.register(Value)
    def numpy_print_value(self, expr):
        return str(expr)
    
    @ExpressionPrinter.__call__.register(ProductBase)
    def numpy_print_product(self, expr):
        return '(%s)' % ' * '.join(map(self, expr.args))
    
    @ExpressionPrinter.__call__.register(SumBase)
    def numpy_print_sum(self, expr):
        return '(%s)' % ' + '.join(map(self, expr.args))
    
    @ExpressionPrinter.__call__.register(QuotientBase)
    def numpy_print_quotient(self, expr):
        return '(%s / %s)' % (self(expr.lhs), self(expr.rhs))
    
    @ExpressionPrinter.__call__.register(DifferenceBase)
    def numpy_print_difference(self, expr):
        return '(%s - %s)' % (self(expr.lhs), self(expr.rhs))
    
    @ExpressionPrinter.__call__.register(GreaterBase)
    def numpy_print_greater(self, expr):
        return '(%s > %s)' % (self(expr.lhs), self(expr.rhs))
    
    @ExpressionPrinter.__call__.register(GreaterEqualBase)
    def numpy_print_greater_equal(self, expr):
        return '(%s >= %s)' % (self(expr.lhs), self(expr.rhs))
    
    @ExpressionPrinter.__call__.register(LessBase)
    def numpy_print_less(self, expr):
        return '(%s < %s)' % (self(expr.lhs), self(expr.rhs))
    
    @ExpressionPrinter.__call__.register(LessEqualBase)
    def numpy_print_less_equal(self, expr):
        return '(%s <= %s)' % (self(expr.lhs), self(expr.rhs))
    
class NumpyPrinter(BasicOperatorPrinter):
    @ExpressionPrinter.__call__.register(RealNumber)
    def numpy_print_real_number(self, expr):
        return repr(expr.value)
    
    @ExpressionPrinter.__call__.register(And)
    def numpy_print_and(self, expr):
        return reduction('logical_and', self, expr.args)
    
    @ExpressionPrinter.__call__.register(Or)
    def numpy_print_or(self, expr):
        return reduction('logical_or', self, expr.args)
    
    @ExpressionPrinter.__call__.register(Not)
    def numpy_print_not(self, expr):
        return 'logical_not(%s)' % self(expr.arg)
    
    @ExpressionPrinter.__call__.register(Variable)
    def numpy_print_variable(self, expr):
        return expr.name
    
    @ExpressionPrinter.__call__.register(WeightedMode)
    def numpy_print_weighted_mode(self, expr):
        return ('apply_along_axis(compose(argmax, partial(bincount, weights=array([%s]))), axis=1, arr=array([%s]).astype(int).T)'
                % 
                (
                 ', '.join(map(self, expr.weights)),
                 ', '.join(map(self, expr.data)),
                 ))
    
    @ExpressionPrinter.__call__.register(FiniteMap)
    def numpy_print_finite_map(self, expr):
        return 'vectorize({%s}.get)(%s)' % (', '.join(map(lambda x: '%s: %s' 
                                                          % (self(x[0]), self(x[1])), expr.mapping.items())),
                                            self(expr.arg))
    
    @ExpressionPrinter.__call__.register(NegateBase)
    def numpy_print_negate(self, expr):
        return '-%s' % self(expr.arg)
    
    @ExpressionPrinter.__call__.register(Log)
    def numpy_print_log(self, expr):
        return 'log(%s)' % self(expr.arg)
    
    @ExpressionPrinter.__call__.register(Expit)
    def numpy_print_expit(self, expr):
        return 'expit(%s)' % self(expr.arg)
    
    @ExpressionPrinter.__call__.register(MaxBase)
    def numpy_print_max(self, expr):
        return 'maximum(%s)' % ', '.join(map(self, expr.args))
    
    @ExpressionPrinter.__call__.register(MinBase)
    def numpy_print_min(self, expr):
        return 'minimum(%s)' % ', '.join(map(self, expr.args))
    
    @ExpressionPrinter.__call__.register(GreaterBase)
    def numpy_print_greater(self, expr):
        return 'greater(%s, %s)' % (self(expr.lhs), self(expr.rhs))
    
    @ExpressionPrinter.__call__.register(GreaterEqualBase)
    def numpy_print_greater_equal(self, expr):
        return 'greater_equal(%s, %s)' % (self(expr.lhs), self(expr.rhs))
    
    @ExpressionPrinter.__call__.register(LessBase)
    def numpy_print_less(self, expr):
        return 'less(%s, %s)' % (self(expr.lhs), self(expr.rhs))
    
    @ExpressionPrinter.__call__.register(LessEqualBase)
    def numpy_print_less_equal(self, expr):
        return 'less_equal(%s, %s)' % (self(expr.lhs), self(expr.rhs))
    
    @ExpressionPrinter.__call__.register(RealPiecewise)
    def numpy_print_real_piecewise(self, expr):
        vals, conds = zip(*expr.pairs)
        vals = '[{0}]'.format(', '.join(self(val) for val in vals))
        conds = '[{0}]'.format(', '.join(self(cond) for cond in conds))
        return 'select({0}, {1}, default=nan).astype(float)'.format(conds, vals)
    
    @ExpressionPrinter.__call__.register(IntegerPiecewise)
    def numpy_print_integer_piecewise(self, expr):
        vals, conds = zip(*expr.pairs)
        vals = '[{0}]'.format(', '.join(self(val) for val in vals))
        conds = '[{0}]'.format(', '.join(self(cond) for cond in conds))
        return 'select({0}, {1}, default=nan).astype(int)'.format(conds, vals)
    
    @ExpressionPrinter.__call__.register(BoolPiecewise)
    def numpy_print_bool_piecewise(self, expr):
        vals, conds = zip(*expr.pairs)
        vals = '[{0}]'.format(', '.join(self(val) for val in vals))
        conds = '[{0}]'.format(', '.join(self(cond) for cond in conds))
        return 'select({0}, {1}, default=nan).astype(bool)'.format(conds, vals)
    
    @ExpressionPrinter.__call__.register(Nan)
    def numpy_print_nan(self, expr):
        return 'nan'
    
    @ExpressionPrinter.__call__.register(IsNan)
    def numpy_print_is_nan(self, expr):
        return 'isnan(%s)' % self(expr.arg)
    
    @ExpressionPrinter.__call__.register(BoolToReal)
    def numpy_print_bool_to_real(self, expr):
        return '%s' % self(expr.arg)

class PandasPrinter(NumpyPrinter):
    @ExpressionPrinter.__call__.register(Variable)
    def pandas_print_variable(self, expr):
        return 'asarray(dataframe[\'' + expr.name + '\'])'

class JavascriptPrinter(BasicOperatorPrinter):
    @ExpressionPrinter.__call__.register(Variable)
    def js_print_variable(self, expr):
        return expr.name
    
    @ExpressionPrinter.__call__.register(MaxBase)
    def js_print_max(self, expr):
        return 'Math.max(' + ','.join(self(arg) for arg in expr.args) + ')'
    
    @ExpressionPrinter.__call__.register(MinBase)
    def js_print_min(self, expr):
        return 'Math.min(' + ','.join(self(arg) for arg in expr.args) + ')'
    
    @ExpressionPrinter.__call__.register(IsNan)
    def js_print_is_nan(self, expr):
        return 'isNan(' + self(expr) + ')'
    
    @ExpressionPrinter.__call__.register(Expit)
    def js_print_expit(self, expr):
        return 'expit(' + self(expr.arg) + ')'
    
    @ExpressionPrinter.__call__.register(Nan)
    def js_print_nan(self, expr):
        return 'NaN'
    
    @ExpressionPrinter.__call__.register(PiecewiseBase)
    def js_print_piecewise(self, expr):
        return (':'.join(map(lambda pair: '(%s?%s' % 
                                  (str(pair[1]), str(pair[0])), expr.pairs)) + 
                (')' * len(expr.pairs)))


