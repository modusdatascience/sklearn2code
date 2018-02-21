from sklearn2code.dispatching import Dispatcher, Rep
from nose.tools import assert_equal

def test_dispatcher():
    f = Dispatcher('f')
    
    def f_int(x):
        return x + 1
    f.register[int](f_int)
    
    def f_int_int(x, y):
        return x + y
    f.register[int:2](f_int_int)
    
    def f_ints(*args):
        return sum(args)
    f.register[int:Rep(0,None)](f_ints)
    
    def f_str(x):
        return x + '1'
    f.register[str](f_str)
    
#     assert_equal(f(1), 2)
#     assert_equal(f(1, 2), 3)
    assert_equal(f(1, 2, 3), 6)
    assert_equal(f('1'), '11')
    
if __name__ == '__main__':
    # This code will run the test in this file.'
    import sys
    import nose
    module_name = sys.modules[__name__].__file__

    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s', '-v'])
