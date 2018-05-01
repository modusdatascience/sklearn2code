from sklearn2code.sym.expression import RealNumber, Log,\
    PiecewiseBase, NegateBase, MaxBase,\
    MinBase, GreaterBase, GreaterEqualBase, LessEqualBase, LessBase, Nan, IsNan,\
    ProductBase, SumBase, QuotientBase, DifferenceBase, Value, Expit, And, Or,\
    Variable, BoolToReal, Not, FiniteMap, WeightedMode, RealPiecewise,\
    IntegerPiecewise, BoolPiecewise, WeightedMedian, EqualsBase, Boolean,\
    VectorExpression
from six import with_metaclass
from toolz.functoolz import curry, flip
from multipledispatch.dispatcher import Dispatcher

class ExpressionTypeNotSupportedError(Exception):
    pass

@curry
def reduction(binary, self, args):
    if len(args) == 1:
        return self(args[0])
    else:
        return binary(self(args[0]), reduction(binary, self, args[1:]))

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
    @ExpressionPrinter.__call__.register(Value, int)
    def basic_print_value(self, expr, depth):
        return str(expr)
    
    @ExpressionPrinter.__call__.register(ProductBase, int)
    def basic_print_product(self, expr, depth):
        return '(%s)' % ' * '.join(map(flip(self)(depth+1), expr.args))
    
    @ExpressionPrinter.__call__.register(SumBase, int)
    def basic_print_sum(self, expr, depth):
        return '(%s)' % ' + '.join(map(flip(self)(depth+1), expr.args))
    
    @ExpressionPrinter.__call__.register(QuotientBase, int)
    def basic_print_quotient(self, expr, depth):
        return '(%s / %s)' % (self(expr.lhs, depth+1), self(expr.rhs, depth+1))
    
    @ExpressionPrinter.__call__.register(DifferenceBase, int)
    def basic_print_difference(self, expr, depth):
        return '(%s - %s)' % (self(expr.lhs, depth+1), self(expr.rhs, depth+1))
    
    @ExpressionPrinter.__call__.register(GreaterBase, int)
    def basic_print_greater(self, expr, depth):
        return '(%s > %s)' % (self(expr.lhs, depth+1), self(expr.rhs, depth+1))
    
    @ExpressionPrinter.__call__.register(GreaterEqualBase, int)
    def basic_print_greater_equal(self, expr, depth):
        return '(%s >= %s)' % (self(expr.lhs, depth+1), self(expr.rhs, depth+1))
    
    @ExpressionPrinter.__call__.register(LessBase, int)
    def basic_print_less(self, expr, depth):
        return '(%s < %s)' % (self(expr.lhs, depth+1), self(expr.rhs, depth+1))
    
    @ExpressionPrinter.__call__.register(LessEqualBase, int)
    def basic_print_less_equal(self, expr, depth):
        return '(%s <= %s)' % (self(expr.lhs, depth+1), self(expr.rhs, depth+1))
    
    @ExpressionPrinter.__call__.register(VectorExpression, int)
    def basic_print_vector_expression(self, expr, depth):
        if len(expr.components) > 1:
            return '[%s]' % (', '.join(map(flip(self)(depth+1), expr.components)))
        elif len(expr.components) == 1:
            # Special case for vector of length 1: treat as scalar.
            # An unfortunate compromise for compatibility with various 
            # languages' multiple assignment syntax.  
            return self(expr.components[0], depth+1)
        else:
            return '[]'
    
class NumpyPrinter(BasicOperatorPrinter):
    @ExpressionPrinter.__call__.register(VectorExpression, int)
    def numpy_print_vector_expression(self, expr, depth):
        return 'array([%s]).T' % (', '.join(map(flip(self)(depth+1), expr.components)))
    
    @ExpressionPrinter.__call__.register(RealNumber, int)
    def numpy_print_real_number(self, expr, depth):
        return repr(expr.value)
    
    @ExpressionPrinter.__call__.register(And, int)
    def numpy_print_and(self, expr, depth):
        return reduction(lambda x,y: 'logical_and(%s, %s)' % (x, y), flip(self)(depth+1), expr.args)
    
    @ExpressionPrinter.__call__.register(Or, int)
    def numpy_print_or(self, expr, depth):
        return reduction(lambda x,y: 'logical_or(%s, %s)' % (x, y), flip(self)(depth+1), expr.args)
#         return reduction('logical_or', self, expr.args)
    
    @ExpressionPrinter.__call__.register(Not, int)
    def numpy_print_not(self, expr, depth):
        return 'logical_not(%s)' % self(expr.arg, depth+1)
    
    @ExpressionPrinter.__call__.register(Variable, int)
    def numpy_print_variable(self, expr, depth):
        return expr.name
    
    @ExpressionPrinter.__call__.register(WeightedMode, int)
    def numpy_print_weighted_mode(self, expr, depth):
        return ('apply_along_axis(compose(argmax, partial(bincount, weights=array([%s]))), axis=1, arr=array([%s]).astype(int).T)'
                % 
                (
                 ', '.join(map(flip(self)(depth+1), expr.weights)),
                 ', '.join(map(flip(self)(depth+1), expr.data)),
                 ))
        
    @ExpressionPrinter.__call__.register(WeightedMedian, int)    
    def numpy_print_weighted_median(self, expr, depth):
        return('weighted_median(weights=array([%s]), data=array([%s]))'
               %
               (
                ', '.join(map(flip(self)(depth+1), expr.weights)), 
                ', '.join(map(flip(self)(depth+1), expr.data)),
                ))
    
    @ExpressionPrinter.__call__.register(FiniteMap, int)
    def numpy_print_finite_map(self, expr, depth):
        return 'vectorize({%s}.get)(%s)' % (', '.join(map(lambda x: '%s: %s' 
                                                          % (self(x[0], depth+1), self(x[1], depth+1)), expr.mapping.items())),
                                            self(expr.arg, depth+1))
    
    @ExpressionPrinter.__call__.register(NegateBase, int)
    def numpy_print_negate(self, expr, depth):
        return '-%s' % self(expr.arg, depth+1)
    
    @ExpressionPrinter.__call__.register(Log, int)
    def numpy_print_log(self, expr, depth):
        return 'log(%s)' % self(expr.arg, depth+1)
    
    @ExpressionPrinter.__call__.register(Expit, int)
    def numpy_print_expit(self, expr, depth):
        return 'expit(%s)' % self(expr.arg, depth+1)
    
    @ExpressionPrinter.__call__.register(MaxBase, int)
    def numpy_print_max(self, expr, depth):
        return 'maximum(%s)' % ', '.join(map(flip(self)(depth+1), expr.args))
    
    @ExpressionPrinter.__call__.register(MinBase, int)
    def numpy_print_min(self, expr, depth):
        return 'minimum(%s)' % ', '.join(map(flip(self)(depth+1), expr.args))
    
    @ExpressionPrinter.__call__.register(GreaterBase, int)
    def numpy_print_greater(self, expr, depth):
        return 'greater(%s, %s)' % (self(expr.lhs, depth+1), self(expr.rhs, depth+1))
    
    @ExpressionPrinter.__call__.register(GreaterEqualBase, int)
    def numpy_print_greater_equal(self, expr, depth):
        return 'greater_equal(%s, %s)' % (self(expr.lhs, depth+1), self(expr.rhs, depth+1))
    
    @ExpressionPrinter.__call__.register(LessBase, int)
    def numpy_print_less(self, expr, depth):
        return 'less(%s, %s)' % (self(expr.lhs, depth+1), self(expr.rhs, depth+1))
    
    @ExpressionPrinter.__call__.register(LessEqualBase, int)
    def numpy_print_less_equal(self, expr, depth):
        return 'less_equal(%s, %s)' % (self(expr.lhs, depth+1), self(expr.rhs, depth+1))
    
    @ExpressionPrinter.__call__.register(RealPiecewise, int)
    def numpy_print_real_piecewise(self, expr, depth):
        vals, conds = zip(*expr.pairs)
        vals = '[{0}]'.format(', '.join(self(val, depth+1) for val in vals))
        conds = '[{0}]'.format(', '.join(self(cond, depth+1) for cond in conds))
        return 'select({0}, {1}, default=nan).astype(float)'.format(conds, vals)
    
    @ExpressionPrinter.__call__.register(IntegerPiecewise, int)
    def numpy_print_integer_piecewise(self, expr, depth):
        vals, conds = zip(*expr.pairs)
        vals = '[{0}]'.format(', '.join(self(val, depth+1) for val in vals))
        conds = '[{0}]'.format(', '.join(self(cond, depth+1) for cond in conds))
        return 'select({0}, {1}, default=nan).astype(int)'.format(conds, vals)
    
    @ExpressionPrinter.__call__.register(BoolPiecewise, int)
    def numpy_print_bool_piecewise(self, expr, depth):
        vals, conds = zip(*expr.pairs)
        vals = '[{0}]'.format(', '.join(self(val, depth+1) for val in vals))
        conds = '[{0}]'.format(', '.join(self(cond, depth+1) for cond in conds))
        return 'select({0}, {1}, default=nan).astype(bool)'.format(conds, vals)
    
    @ExpressionPrinter.__call__.register(Nan, int)
    def numpy_print_nan(self, expr, depth):
        return 'nan'
    
    @ExpressionPrinter.__call__.register(IsNan, int)
    def numpy_print_is_nan(self, expr, depth):
        return 'isnan(%s)' % self(expr.arg, depth+1)
    
    @ExpressionPrinter.__call__.register(BoolToReal, int)
    def numpy_print_bool_to_real(self, expr, depth):
        return '%s' % self(expr.arg, depth+1)

class PandasPrinter(NumpyPrinter):
    @ExpressionPrinter.__call__.register(VectorExpression, int)
    def pandas_print_vector_expression(self, expr, depth):
        raise ExpressionTypeNotSupportedError('Because pandas is a flat data format, it\'s not possible to render vector expressions.')
    
    @ExpressionPrinter.__call__.register(Variable, int)
    def pandas_print_variable(self, expr, depth):
        return 'asarray(dataframe[\'' + expr.name + '\'])'

class JavascriptPrinter(BasicOperatorPrinter):
    @ExpressionPrinter.__call__.register(VectorExpression, int)
    def js_print_vector_expression(self, expr, depth):
        raise ExpressionTypeNotSupportedError('Because pandas is a flat data format, it\'s not possible to render vector expressions.')
    
    @ExpressionPrinter.__call__.register(Boolean, int)
    def js_print_boolean(self, expr, depth):
        return 'true' if expr.value == True else 'false' if expr.value == False else NotImplemented
    
    @ExpressionPrinter.__call__.register(Variable, int)
    def js_print_variable(self, expr, depth):
        return expr.name
    
    @ExpressionPrinter.__call__.register(MaxBase, int)
    def js_print_max(self, expr, depth):
        return 'Math.max(' + ','.join(self(arg, depth+1) for arg in expr.args) + ')'
    
    @ExpressionPrinter.__call__.register(MinBase, int)
    def js_print_min(self, expr, depth):
        return 'Math.min(' + ','.join(self(arg, depth+1) for arg in expr.args) + ')'
    
    @ExpressionPrinter.__call__.register(IsNan, int)
    def js_print_is_nan(self, expr, depth):
        return 'isNaN(' + self(expr.arg, depth+1) + ')'
    
    @ExpressionPrinter.__call__.register(Expit, int)
    def js_print_expit(self, expr, depth):
        return 'expit(' + self(expr.arg, depth+1) + ')'
    
    @ExpressionPrinter.__call__.register(Nan, int)
    def js_print_nan(self, expr, depth):
        return 'NaN'
    
    @ExpressionPrinter.__call__.register(PiecewiseBase, int)
    def js_print_piecewise(self, expr, depth):
        return (':'.join(map(lambda pair: '(%s?%s' % 
                                  (self(pair[1], depth+1), self(pair[0], depth+1)), expr.pairs)) + 
                ':null' + (')' * len(expr.pairs)))
    
    @ExpressionPrinter.__call__.register(BoolToReal, int)
    def js_print_bool_to_real(self, expr, depth):
        return '(%s==true?1.0:0.0)' % self(expr.arg, depth+1)
    
    @ExpressionPrinter.__call__.register(WeightedMode, int)
    def js_print_weighted_mode(self, expr, depth):
        return 'weightedMode([%s], [%s])' % (', '.join(self(x, depth+1) for x in expr.data), 
                                             ', '.join(self(x, depth+1) for x in expr.weights)
                                             )
    
    @ExpressionPrinter.__call__.register(WeightedMedian, int)
    def js_print_weighted_median(self, expr, depth):
        return 'weightedMedian([%s], [%s])' % (', '.join(self(x, depth+1) for x in expr.data), 
                                             ', '.join(self(x, depth+1) for x in expr.weights)
                                             )

    @ExpressionPrinter.__call__.register(Not, int)
    def js_print_not(self, expr, depth):
        return '!(%s)' % self(expr.arg, depth+1)
    
    @ExpressionPrinter.__call__.register(Or, int)
    def js_print_or(self, expr, depth):
        return reduction(lambda x, y: '(%s || %s)' % (x, y), flip(self)(depth+1), expr.args)
#         return '(%s || %s)' % (self(expr.lhs), self(expr.rhs))
    
    @ExpressionPrinter.__call__.register(And, int)
    def js_print_and(self, expr, depth):
        return reduction(lambda x, y: '(%s && %s)' % (x, y), flip(self)(depth+1), expr.args)
    
    @ExpressionPrinter.__call__.register(EqualsBase, int)
    def js_print_equals(self, expr, depth):
        return '(%s === %s)' % (self(expr.lhs, depth+1), self(expr.rhs, depth+1))
    
    @ExpressionPrinter.__call__.register(FiniteMap, int)
    def js_print_finite_map(self, expr, depth):
        arg = self(expr.arg, depth+1)
        return (':'.join(map(lambda pair: '(%s===%s?%s' % 
                                  (arg, self(pair[0], depth+1), self(pair[1], depth+1)), expr.mapping.items())) + 
                ':null' + ')' * (len(expr.mapping)))
