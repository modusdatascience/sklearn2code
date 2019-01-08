from .expression import RealNumber, Log,\
    PiecewiseBase, NegateBase, MaxBase,\
    MinBase, GreaterBase, GreaterEqualBase, LessEqualBase, LessBase, Nan, IsNan,\
    ProductBase, SumBase, QuotientBase, DifferenceBase, Value, Expit, And, Or,\
    Variable, BoolToReal, Not, FiniteMap, WeightedMode, RealPiecewise,\
    IntegerPiecewise, BoolPiecewise, WeightedMedian, EqualsBase, Boolean,\
    VectorExpression
from six import with_metaclass
from toolz.functoolz import curry
from multipledispatch.dispatcher import Dispatcher
from ..utility import HeritableDispatcherMeta, HeritableDispatcher

class ExpressionTypeNotSupportedError(Exception):
    pass

@curry
def reduction(binary, self, args):
    if len(args) == 1:
        return self(args[0])
    else:
        return binary(self(args[0]), reduction(binary, self, args[1:]))

class ExpressionPrinter(with_metaclass(HeritableDispatcherMeta, object)):
    __call__ = HeritableDispatcher('__call__')

class BasicOperatorPrinter(ExpressionPrinter):
    @ExpressionPrinter.__call__.register(Value)
    def basic_print_value(self, expr):
        return str(expr)
    
    @ExpressionPrinter.__call__.register(ProductBase)
    def basic_print_product(self, expr):
        return '(%s)' % ' * '.join(map(self, expr.args))
    
    @ExpressionPrinter.__call__.register(SumBase)
    def basic_print_sum(self, expr):
        return '(%s)' % ' + '.join(map(self, expr.args))
    
    @ExpressionPrinter.__call__.register(QuotientBase)
    def basic_print_quotient(self, expr):
        return '(%s / %s)' % (self(expr.lhs), self(expr.rhs))
    
    @ExpressionPrinter.__call__.register(DifferenceBase)
    def basic_print_difference(self, expr):
        return '(%s - %s)' % (self(expr.lhs), self(expr.rhs))
    
    @ExpressionPrinter.__call__.register(GreaterBase)
    def basic_print_greater(self, expr):
        return '(%s > %s)' % (self(expr.lhs), self(expr.rhs))
    
    @ExpressionPrinter.__call__.register(GreaterEqualBase)
    def basic_print_greater_equal(self, expr):
        return '(%s >= %s)' % (self(expr.lhs), self(expr.rhs))
    
    @ExpressionPrinter.__call__.register(LessBase)
    def basic_print_less(self, expr):
        return '(%s < %s)' % (self(expr.lhs), self(expr.rhs))
    
    @ExpressionPrinter.__call__.register(LessEqualBase)
    def basic_print_less_equal(self, expr):
        return '(%s <= %s)' % (self(expr.lhs), self(expr.rhs))
    
    @ExpressionPrinter.__call__.register(VectorExpression)
    def basic_print_vector_expression(self, expr):
        if len(expr.components) > 1:
            return '[%s]' % (', '.join(map(self, expr.components)))
        elif len(expr.components) == 1:
            # Special case for vector of length 1: treat as scalar.
            # An unfortunate compromise for compatibility with various 
            # languages' multiple assignment syntax.  
            return self(expr.components[0])
        else:
            return '[]'
    
class NumpyPrinter(BasicOperatorPrinter):
    @ExpressionPrinter.__call__.register(VectorExpression)
    def numpy_print_vector_expression(self, expr):
        return 'array([%s]).T' % (', '.join(map(self, expr.components)))
    
    @ExpressionPrinter.__call__.register(RealNumber)
    def numpy_print_real_number(self, expr):
        return repr(expr.value)
    
    @ExpressionPrinter.__call__.register(And)
    def numpy_print_and(self, expr):
        return reduction(lambda x,y: 'logical_and(%s, %s)' % (x, y), self, expr.args)
    
    @ExpressionPrinter.__call__.register(Or)
    def numpy_print_or(self, expr):
        return reduction(lambda x,y: 'logical_or(%s, %s)' % (x, y), self, expr.args)
    
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
        
    @ExpressionPrinter.__call__.register(WeightedMedian)    
    def numpy_print_weighted_median(self, expr):
        return('weighted_median(weights=array([%s]), data=array([%s]))'
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
    @ExpressionPrinter.__call__.register(VectorExpression)
    def pandas_print_vector_expression(self, expr):
        raise ExpressionTypeNotSupportedError('Because pandas is a flat data format, it\'s not possible to render vector expressions.')
    
    @ExpressionPrinter.__call__.register(Variable)
    def pandas_print_variable(self, expr):
        return 'asarray(dataframe[\'' + expr.name + '\'])'

class JavascriptPrinter(BasicOperatorPrinter):
    @ExpressionPrinter.__call__.register(VectorExpression)
    def js_print_vector_expression(self, expr):
        raise ExpressionTypeNotSupportedError('Because pandas is a flat data format, it\'s not possible to render vector expressions.')
    
    @ExpressionPrinter.__call__.register(Boolean)
    def js_print_boolean(self, expr):
        return 'true' if expr.value == True else 'false' if expr.value == False else NotImplemented
    
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
        return 'isNaN(' + self(expr.arg) + ')'
    
    @ExpressionPrinter.__call__.register(Expit)
    def js_print_expit(self, expr):
        return 'expit(' + self(expr.arg) + ')'
    
    @ExpressionPrinter.__call__.register(Nan)
    def js_print_nan(self, expr):
        return 'NaN'
    
    @ExpressionPrinter.__call__.register(PiecewiseBase)
    def js_print_piecewise(self, expr):
        return (':'.join(map(lambda pair: '(%s?%s' % 
                                  (self(pair[1]), self(pair[0])), expr.pairs)) + 
                ':null' + (')' * len(expr.pairs)))
    
    @ExpressionPrinter.__call__.register(BoolToReal)
    def js_print_bool_to_real(self, expr):
        return '(%s==true?1.0:0.0)' % self(expr.arg)
    
    @ExpressionPrinter.__call__.register(WeightedMode)
    def js_print_weighted_mode(self, expr):
        return 'weightedMode([%s], [%s])' % (', '.join(self(x) for x in expr.data), 
                                             ', '.join(self(x) for x in expr.weights)
                                             )
    
    @ExpressionPrinter.__call__.register(WeightedMedian)
    def js_print_weighted_median(self, expr):
        return 'weightedMedian([%s], [%s])' % (', '.join(self(x) for x in expr.data), 
                                             ', '.join(self(x) for x in expr.weights)
                                             )

    @ExpressionPrinter.__call__.register(Not)
    def js_print_not(self, expr):
        return '!(%s)' % self(expr.arg)
    
    @ExpressionPrinter.__call__.register(Or)
    def js_print_or(self, expr):
        return reduction(lambda x, y: '(%s || %s)' % (x, y), self, expr.args)
#         return '(%s || %s)' % (self(expr.lhs), self(expr.rhs))
    
    @ExpressionPrinter.__call__.register(And)
    def js_print_and(self, expr):
        return reduction(lambda x, y: '(%s && %s)' % (x, y), self, expr.args)
    
    @ExpressionPrinter.__call__.register(EqualsBase)
    def js_print_equals(self, expr):
        return '(%s === %s)' % (self(expr.lhs), self(expr.rhs))
    
    @ExpressionPrinter.__call__.register(FiniteMap)
    def js_print_finite_map(self, expr):
        arg = self(expr.arg)
        return (':'.join(map(lambda pair: '(%s===%s?%s' % 
                                  (arg, self(pair[0]), self(pair[1])), expr.mapping.items())) + 
                ':null' + ')' * (len(expr.mapping)))
