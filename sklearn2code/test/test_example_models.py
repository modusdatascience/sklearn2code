import numpy as np
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.base import clone
from numpy.ma.testutils import assert_array_almost_equal
from six import PY2
from pandas import DataFrame
from sklearn2code.languages import numpy_flat, pandas
from sklearn2code.sklearn2code import sklearn2code
from sklearn2code.utility import exec_module
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from pyearth.earth import Earth
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble.voting_classifier import VotingClassifier
from sklearn.linear_model.coordinate_descent import Lasso, ElasticNet,\
    ElasticNetCV, LassoCV
from sklearn.linear_model.ridge import Ridge, RidgeCV
from sklearn.linear_model.stochastic_gradient import SGDRegressor
from sklearn.ensemble.weight_boosting import AdaBoostRegressor

if PY2:
    from types import MethodType
    
def create_weird_classification_problem_1(m=1000, n=10):
    np.random.seed(1)
    X = DataFrame(np.random.normal(size=(m,n)), columns=['x%d' % i for i in range(n)])
    thresh = np.random.normal(size=n)
    X_transformed = X * (X > thresh)
    beta = np.random.normal(size=n)
    y = (np.dot(X_transformed, beta) + np.random.normal(size=m)) > 0
    return (dict(X=X, y=y), dict(X=X), dict(X=X))

def create_isotonic_regression_problem_1(m=1000):
    np.random.seed(1)
    X = np.random.normal(size=m)
    y = np.random.normal(X)
    return (dict(X=X, y=y), dict(T=X), dict(X=DataFrame(dict(x=X))))

def create_regression_problem_1(m=1000, n=10):
    np.random.seed(1)
    X = DataFrame(np.random.normal(size=(m,n)), columns=['x%d' % i for i in range(n)])
    beta = np.random.normal(size=n) *10
    y = np.random.normal(np.dot(X, beta), .1)
    return (dict(X=X, y=y), dict(X=X), dict(X=X))

def create_regression_problem_with_missingness_1(m=1000, n=10):
    np.random.seed(1)
    X = np.random.normal(size=(m,n))
    beta = np.random.normal(size=n)
    y = np.random.normal(np.dot(X, beta))
    missing = np.random.binomial(1, .1, size=X.shape)
    X[missing] = np.nan
    X = DataFrame(X, columns=['x%d' % i for i in range(n)])
    return (dict(X=X, y=y), dict(X=X), dict(X=X))

test_cases = [
#             (VotingClassifier([('logistic', LogisticRegression()), ('earth', Pipeline([('earth', Earth()), ('logistic', LogisticRegression())]))]), 
#              ['predict'], create_weird_classification_problem_1()),
#             (GradientBoostingClassifier(max_depth=10, n_estimators=10), ['predict_proba', 'predict'], create_weird_classification_problem_1()),
            (AdaBoostRegressor(), ['predict'], create_regression_problem_1()), 
#             (LogisticRegression(), ['predict_proba', 'predict'], create_weird_classification_problem_1()),
#             (IsotonicRegression(out_of_bounds='clip'), ['predict'], create_isotonic_regression_problem_1()),
#             (Earth(), ['predict', 'transform'], create_regression_problem_1()),
#             (Earth(allow_missing=True), ['predict', 'transform'], create_regression_problem_with_missingness_1()),
#             (ElasticNet(), ['predict'], create_regression_problem_1()),
#             (ElasticNetCV(), ['predict'], create_regression_problem_1()),
#             (LassoCV(), ['predict'], create_regression_problem_1()),
#             (Ridge(), ['predict'], create_regression_problem_1()), 
#             (RidgeCV(), ['predict'], create_regression_problem_1()), 
#             (SGDRegressor(), ['predict'], create_regression_problem_1()),
#             (Lasso(), ['predict'], create_regression_problem_1()),
#             (Pipeline([('earth', Earth()), ('logistic', LogisticRegression())]), ['predict', 'predict_proba'], create_weird_classification_problem_1()),
#             (FeatureUnion([('earth', Earth()), ('earth2', Earth(max_degree=2))], transformer_weights={'earth':1, 'earth2':2}),
#             ['transform'], create_weird_classification_problem_1())
              ]

# Create tests for numpy_flat language
def create_case_numpy_flat(estimator, methods, fit_data, predict_data, export_predict_data):
    def test_case(self):
        model = clone(estimator)
        model.fit(**fit_data)
        
        for method in  methods:
            pred = getattr(model, method)(**predict_data)
            code = sklearn2code(model, method, numpy_flat)
            module = exec_module('test_module', code)
            exported_pred = getattr(module, method)(**export_predict_data['X'])
            if isinstance(exported_pred, tuple):
                exported_pred = DataFrame(dict(enumerate(exported_pred)))
            assert_array_almost_equal(pred, exported_pred)
    test_case.__doc__ = ('Testing numpy_flat language exportability of method%s %s of %s' % 
                         ('s' if len(methods)>1 else '', ', '.join(methods), repr(estimator)))
    return test_case

# All tests will be methods of this class
class TestExampleEstimatorsNumpyFlat(object):
    pass


# The following loop adds a method to TestExampleEstimators for each test case
for i, (estimator, methods, (fit_data, predict_data, export_predict_data)) in enumerate(test_cases):
    case = create_case_numpy_flat(estimator, methods, fit_data, predict_data, export_predict_data)
    case_name = 'test_case_%d' % i
    case.__name__ = case_name
    if PY2:
        case = MethodType(case, None, TestExampleEstimatorsNumpyFlat)
    setattr(TestExampleEstimatorsNumpyFlat, case_name, case)
    del case

# Create tests for pandas language
def create_case_pandas(estimator, methods, fit_data, predict_data, export_predict_data):
    def test_case(self):
        model = clone(estimator)
        model.fit(**fit_data)
        
        for method in  methods:
            pred = DataFrame(getattr(model, method)(**predict_data))
            code = sklearn2code(model, method, pandas)
            module = exec_module('test_module', code)
            exported_pred = getattr(module, method)(export_predict_data['X'])
            assert_array_almost_equal(pred, exported_pred)
    test_case.__doc__ = ('Testing pandas language exportability of method%s %s of %s' % 
                         ('s' if len(methods)>1 else '', ', '.join(methods), repr(estimator)))
    return test_case

# All tests will be methods of this class
class TestExampleEstimatorsPandas(object):
    pass

# The following loop adds a method to TestExampleEstimators for each test case
for i, (estimator, methods, (fit_data, predict_data, export_predict_data)) in enumerate(test_cases):
    case = create_case_pandas(estimator, methods, fit_data, predict_data, export_predict_data)
    case_name = 'test_case_%d' % i
    case.__name__ = case_name
    if PY2:
        case = MethodType(case, None, TestExampleEstimatorsPandas)
    setattr(TestExampleEstimatorsPandas, case_name, case)
    del case




if __name__ == '__main__':
    # This code will run the test in this file.'
    import sys
    import nose
    module_name = sys.modules[__name__].__file__

    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s', '-v'])

    
    