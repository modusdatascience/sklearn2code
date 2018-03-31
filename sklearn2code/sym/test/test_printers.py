from sklearn2code.sym.expression import FiniteMap, Integer, false, true,\
    IntegerVariable, RealPiecewise, RealNumber
from sklearn2code.sym.printers import JavascriptPrinter
from nose.tools import assert_equal


def test_javascript_finite_map():
    expr = FiniteMap({Integer(0): false, Integer(1): true}, IntegerVariable('x'))
    assert_equal(JavascriptPrinter()(expr), '(x===0?false:(x===1?true:null))')

def test_javascript_piecewise():
    expr = RealPiecewise((RealNumber(0), false), (RealNumber(1), true))
    assert_equal(JavascriptPrinter()(expr), '(false?0.0:(true?1.0:null))')

if __name__ == '__main__':
    import sys
    import nose
    # This code will run the test in this file.'
    module_name = sys.modules[__name__].__file__
 
    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s', '-v'])

