'''
Generate and test all cases using the numpy_flat language.
'''
import numpy as np
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.base import clone
from toolz.dicttoolz import merge
from numpy.ma.testutils import assert_array_almost_equal
from six import PY2, PY3
from pandas import DataFrame
from sklearn2code.languages import numpy_flat
from sklearn2code.sklearn2code import sklearn2code
from sklearn2code.utilty import exec_module

def create_weird_classification_problem_1(m=1000, n=10):
    X = DataFrame(np.random.normal(size=(m,n)), columns=['x%d' % i for i in range(n)])
    thresh = np.random.normal(size=n)
    X_transformed = X * (X > thresh)
    beta = np.random.normal(size=n)
    y = (np.dot(X_transformed, beta) + np.random.normal(size=m)) > 0
    return (dict(X=X), dict(y=y))

test_cases = [
              (GradientBoostingClassifier(max_depth=10, n_estimators=10), ['predict_proba'], create_weird_classification_problem_1()),
              ]

# Create tests
def create_case(estimator, methods, predictor_data, response_data):
    def test_case(self):
        model = clone(estimator)
        model.fit(**merge(predictor_data, response_data))
        
        for method in  methods:
            pred = getattr(model, method)(**predictor_data)
            code = sklearn2code(model, method, numpy_flat)
            module = exec_module('test_module', code)
            exported_pred = getattr(module, method)(**predictor_data['X'])
            if isinstance(exported_pred, tuple):
                exported_pred = DataFrame(dict(enumerate(exported_pred)))
            assert_array_almost_equal(pred, exported_pred)
    test_case.__doc__ = ('Testing exportability of method%s %s of %s' % 
                         ('s' if len(methods)>1 else '', ', '.join(methods), repr(estimator)))
    return test_case

# All tests will be methods of this class
class TestExampleEstimators(object):
    pass

if PY2:
    from types import MethodType

# The following loop adds a method to TestExampleEstimators for each test case
for i, (estimator, methods, (predictor_data, response_data)) in enumerate(test_cases):
    case = create_case(estimator, methods, predictor_data, response_data)
    case_name = 'test_case_%d' % i
    case.__name__ = case_name
    if PY2:
        case = MethodType(case, None, TestExampleEstimators)
    setattr(TestExampleEstimators, case_name, case)
    del case

if __name__ == '__main__':
    # This code will run the test in this file.'
    import sys
    import nose
    module_name = sys.modules[__name__].__file__

    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s', '-v'])

    
    