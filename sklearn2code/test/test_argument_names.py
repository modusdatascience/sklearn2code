from sklearn.datasets.base import load_boston
from pandas.core.frame import DataFrame
from sklearn2code.languages import numpy_flat
from sklearn2code.sklearn2code import sklearn2code
from sklearn2code.utility import exec_module
from numpy.testing.utils import assert_array_almost_equal
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor

def test_argument_names():
    boston = load_boston()
    X = DataFrame(boston['data'], columns=boston['feature_names'])
    y = boston['target']
    model = GradientBoostingRegressor(verbose=True).fit(X, y)
    code = sklearn2code(model, ['predict'], numpy_flat, argument_names=X.columns)
    boston_housing_module = exec_module('boston_housing_module', code)
    assert_array_almost_equal(model.predict(X), 
                              boston_housing_module.predict(**X))
    
if __name__ == '__main__':
    # This code will run the test in this file.'
    import sys
    import nose
    module_name = sys.modules[__name__].__file__

    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s', '-v'])

    

