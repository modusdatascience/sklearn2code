from sklearn2code.sym.function import Function
from sympy.core.symbol import Symbol
from nose.tools import assert_list_equal, assert_equal
from operator import __add__, __mul__, __sub__
from six import PY3

def test_map_symbols():
    fun0 = Function(('x', 'y'), tuple(), (Symbol('x') + Symbol('y'),))
    fun = Function(('x', 'y'), (((('z',), (fun0, ('x','y')))),), (Symbol('x') / Symbol('z'),))
    mapped_fun = fun.map_symbols({'x': 'q'})
    assert_list_equal(list(mapped_fun.inputs), list(map(Symbol, ('q', 'y'))))
    assert_equal(set(mapped_fun.calls[0][1][1]), set(map(Symbol, ('q', 'y'))))
    assert_equal(mapped_fun.outputs[0], Symbol('q') / Symbol('z'))

def test_compose():
    fun0 = Function('x', tuple(), (Symbol('x'), 1 - Symbol('x')))
    fun = Function(('x', 'y'), tuple(), (Symbol('x') / Symbol('y'),))
    composed_fun = fun.compose(fun0)
    assert_equal(composed_fun.calls[0][1][0], fun0)
    assert_equal(composed_fun.inputs, fun0.inputs)
    assert_equal(fun.outputs, composed_fun.map_output_symbols(dict(zip(composed_fun.calls[0][0], fun.inputs))))

def test_from_expressions():
    fun = Function.from_expressions((Symbol('x'), Symbol('x') + Symbol('y')))
    assert_equal(fun, Function(('x', 'y'), tuple(), (Symbol('x'), Symbol('x') + Symbol('y'))))

def test_trim():
    fun0 = Function('x', ((('u',), (Function.from_expression(Symbol('x0') + Symbol('x1')), ('x', 'x'))),), 
                    (Symbol('u'), 1 - Symbol('x')))
    fun = Function(('x', 'y'), ((('z','w'), (fun0, ('y',))),), (Symbol('x') / Symbol('w'),)).trim()
    assert_equal(fun.inputs, (Symbol('x'), Symbol('y')))
    assert_equal(fun.outputs, (Symbol('x') / Symbol('w'),))
    assert_equal(fun.calls, (((Symbol('w'),), (Function(('x', ), tuple(), (1-Symbol('x'),)), (Symbol('y'),))),))
    
class TestOps(object):
    pass

def add_op(op):
    def test_op(self):
        fun0 = Function(('x', 'y'), tuple(), (Symbol('x') + Symbol('y'),))
        fun = Function(('x', 'y'), (((('z',), (fun0, ('x','y')))),), (Symbol('x') / Symbol('z'),))
        fun_op_two = op(fun, 2)
        assert_equal(fun_op_two.outputs[0], op(Symbol('x') / Symbol('z'), 2))
        two_op_fun = op(2, fun)
        assert_equal(two_op_fun.outputs[0], op(2, Symbol('x') / Symbol('z')))
        fun_op_fun = op(fun, fun)
        assert_equal(fun_op_fun.outputs[0], op(Symbol('x') / Symbol('z'), Symbol('x') / Symbol('z')))
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
