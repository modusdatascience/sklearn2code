from sklearn2code.sym.expression import true, RealVariable,\
    Piecewise, RealPiecewise, Log, Max, MaxReal, VectorExpression, RealNumber
from nose.tools import assert_equal, assert_is_instance
from sklearn2code.sym.printers import NumpyPrinter


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
