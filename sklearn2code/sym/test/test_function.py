from sklearn2code.sym.function import Function
from nose.tools import assert_list_equal, assert_equal
from operator import __add__, __mul__, __sub__
from six import PY3
from sklearn2code.sym.expression import RealVariable, RealNumber

def test_map_symbols():
    fun0 = Function(('x', 'y'), tuple(), (RealVariable('x') + RealVariable('y'),))
    fun = Function(('x', 'y'), (((('z',), (fun0, ('x','y')))),), (RealVariable('x') / RealVariable('z'),))
    mapped_fun = fun.map_symbols({'x': 'q'})
    assert_list_equal(list(mapped_fun.inputs), list(map(RealVariable, ('q', 'y'))))
    assert_equal(set(mapped_fun.calls[0][1][1]), set(map(RealVariable, ('q', 'y'))))
    assert_equal(mapped_fun.outputs[0], RealVariable('q') / RealVariable('z'))

def test_compose():
    fun0 = Function('x', tuple(), (RealVariable('x'), RealNumber(1) - RealVariable('x')))
    fun = Function(('x', 'y'), tuple(), (RealVariable('x') / RealVariable('y'),))
    composed_fun = fun.compose(fun0)
    assert_equal(composed_fun.calls[0][1][0], fun0)
    assert_equal(composed_fun.inputs, fun0.inputs)
    assert_equal(fun.outputs, composed_fun.map_output_symbols(dict(zip(composed_fun.calls[0][0], fun.inputs))))

def test_from_expressions():
    fun = Function.from_expressions((RealVariable('x'), RealVariable('x') + RealVariable('y')))
    assert_equal(fun, Function(('x', 'y'), tuple(), (RealVariable('x'), RealVariable('x') + RealVariable('y'))))

def test_trim():
    fun0 = Function('x', ((('u',), (Function.from_expression(RealVariable('x0') + RealVariable('x1')), ('x', 'x'))),), 
                    (RealVariable('u'), RealNumber(1) - RealVariable('x')))
    fun = Function(('x', 'y'), ((('z','w'), (fun0, ('y',))),), (RealVariable('x') / RealVariable('w'),)).trim()
    assert_equal(fun.inputs, (RealVariable('x'), RealVariable('y')))
    assert_equal(fun.outputs, (RealVariable('x') / RealVariable('w'),))
    assert_equal(fun.calls, (((RealVariable('w'),), (Function(('x', ), tuple(), (RealNumber(1)-RealVariable('x'),)), (RealVariable('y'),))),))
    
class TestOps(object):
    pass

def add_op(op):
    def test_op(self):
        fun0 = Function(('x', 'y'), tuple(), (RealVariable('x') + RealVariable('y'),))
        fun = Function(('x', 'y'), (((('z',), (fun0, ('x','y')))),), (RealVariable('x') / RealVariable('z'),))
        fun_op_two = op(fun, RealNumber(2))
        assert_equal(fun_op_two.outputs[0], op(RealVariable('x') / RealVariable('z'), RealNumber(2)))
        two_op_fun = op(RealNumber(2), fun)
        assert_equal(two_op_fun.outputs[0], op(RealNumber(2), RealVariable('x') / RealVariable('z')))
        fun_op_fun = op(fun, fun)
        assert_equal(fun_op_fun.outputs[0], op(RealVariable('x') / RealVariable('z'), RealVariable('x') / RealVariable('z')))
        assert_equal(fun_op_fun.inputs, fun.inputs)
        assert_equal(fun_op_fun.calls, fun.calls)
    test_name = 'test_%s' % op.__name__.strip('__')
    test_op.__name__ = test_name
    setattr(TestOps, test_name, test_op)

add_op(__add__)
add_op(__mul__)
add_op(__sub__)
if PY3:
    from operator import __truediv__  # @UnresolvedImport
    add_op(__truediv__)
else:
    from operator import __div__  # @UnresolvedImport
    add_op(__div__)
    
if __name__ == '__main__':
    # This code will run the test in this file.'
    import sys
    import nose
    module_name = sys.modules[__name__].__file__

    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s', '-v'])
