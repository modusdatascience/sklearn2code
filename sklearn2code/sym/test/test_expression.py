from sklearn2code.sym.expression import true, RealVariable,\
    Piecewise, RealPiecewise, Log, Max, MaxReal, RealNumber,\
    Integer, Ordered, Tuple, TupleVariable, IntegerVariable, BooleanVariable
from nose.tools import assert_equal, assert_is_instance
from sklearn2code.sym.printers import NumpyPrinter
from operator import __add__, __truediv__
from toolz.functoolz import flip, curry


def test_piecewise():
    x = RealVariable('x')
    y = RealVariable('y')
    expr = Piecewise((-Log(x), x > y), (y, true))
    assert_equal(str(expr), '(-Log(x) if (x > y) else (y if True))')
    assert_is_instance(expr, RealPiecewise)

def test_max():
    x = RealVariable('x')
    y = RealVariable('y')
    expr = Max(x, y)
    assert_equal(str(expr), 'Max(x, y)')
    assert_is_instance(expr, MaxReal)

def test_numpy_printer():
    x = RealVariable('x')
    y = RealVariable('y')
    numpy_print = NumpyPrinter()
    expr = Piecewise((-Log(x), x > y), (y, true))
    assert_equal(numpy_print(expr), 'select([greater(x, y), True], [-log(x), y], default=nan).astype(float)')

def test_tuple():
    x = RealVariable('x')
    y = RealVariable('y')
    vector = Tuple(x,y)
    assert_equal(vector.dim, 2)

def test_ordered():
    x = RealVariable('x')
    y = RealVariable('y')
    ordered = Ordered(Tuple(x,y))
    assert_equal(ordered.dim, 2)

# def test_vector_add():
#     x = RealVariable('x')
#     y = RealVariable('y')
#     vector = Vector(x,y)
#     assert_equal(vector + RealNumber(1), Vector(*map(flip(__add__)(RealNumber(1)), vector.args)))
#     assert_equal(RealNumber(1) + vector, Vector(*map(curry(__add__)(RealNumber(1)), vector.args)))
#     vector2 = Vector(RealNumber(1.), RealNumber(1.))
#     assert_equal(vector + vector2, vector + RealNumber(1))
#     assert_equal(vector2 + vector, RealNumber(1) + vector)
    
# def test_vector_truediv():
#     x = RealVariable('x')
#     y = RealVariable('y')
#     vector = Vector(x,y)
#     assert_equal(vector / RealNumber(1), Vector(*map(flip(__truediv__)(RealNumber(1)), vector.args)))
#     assert_equal(RealNumber(1) / vector, Vector(*map(curry(__truediv__)(RealNumber(1)), vector.args)))
#     vector2 = Vector(RealNumber(1.), RealNumber(1.))
#     assert_equal(vector / vector2, vector / RealNumber(1))
#     assert_equal(vector2 / vector, RealNumber(1) / vector)

def test_varfactory():
    cases = [
             (RealVariable('y'), RealVariable('x')),
             (IntegerVariable('y'), IntegerVariable('x')),
             (BooleanVariable('y'), BooleanVariable('x')),
             (Ordered(Tuple(RealNumber(2), RealVariable('y'))), TupleVariable('x', 2)),
             
             ]
    
    for expr, var in cases:
        assert_equal(expr.varfactory()('x'), var)

# def test_vector_expression():
#     x = RealVariable('x')
#     v1 = VectorExpression(RealNumber(1), x)
#     v2 = VectorExpression(x, RealNumber(2))
#     assert_equal(v1 + v2, 
#                  VectorExpression(RealNumber(1) + x, x + RealNumber(2)))

if __name__ == '__main__':
    # This code will run the test in this file.'
    import sys
    import nose
    module_name = sys.modules[__name__].__file__

    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s', '-v'])
