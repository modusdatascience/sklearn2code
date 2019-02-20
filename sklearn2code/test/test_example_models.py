import numpy as np
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier,\
    GradientBoostingRegressor
from sklearn.base import clone
from numpy.ma.testutils import assert_array_almost_equal
from six import PY2
from pandas import DataFrame
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
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
import execjs
from sklearn.ensemble.weight_boosting import AdaBoostRegressor
from sklearn.ensemble.bagging import BaggingRegressor, BaggingClassifier
from sklearn2code.sym.printers import ExpressionTypeNotSupportedError
from sklearn.datasets.base import load_boston
from xgboost.sklearn import XGBRegressor
from sklearn2code.renderers import numpy as numpy_renderer

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

def create_regression_problem_for_xgb_1(m=1000, n=10):
    np.random.seed(1)
    X = DataFrame(np.random.normal(size=(m,n)), columns=['x%d' % i for i in range(n)])
    beta = np.random.normal(size=n) *10
    y = np.random.normal(np.dot(X, beta), .1)
    return (dict(X=X, y=y), dict(data=X), dict(X=X))

def create_regression_problem_with_missingness_1(m=1000, n=10):
    np.random.seed(1)
    X = np.random.normal(size=(m,n))
    beta = np.random.normal(size=n)
    y = np.random.normal(np.dot(X, beta))
    missing = np.random.binomial(1, .1, size=X.shape)
    X[missing] = np.nan
    X = DataFrame(X, columns=['x%d' % i for i in range(n)])
    return (dict(X=X, y=y), dict(X=X), dict(X=X))

def create_boston_housing():
    X, y = load_boston(return_X_y=True)
    X = DataFrame(X, columns=['x%d' % i for i in range(X.shape[1])])
    return (dict(X=X, y=y), dict(X=X), dict(X=X))

test_cases = [
#             (VotingClassifier([('logistic', LogisticRegression()), ('earth', Pipeline([('earth', Earth()), ('logistic', LogisticRegression())]))], 'hard', weights=[1.01,1.01]), 
#              ['predict'], create_weird_classification_problem_1()),
#             (GradientBoostingClassifier(max_depth=10, n_estimators=10), ['predict_proba', 'predict'], create_weird_classification_problem_1()),
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
#             ['transform'], create_weird_classification_problem_1()),
#             (RandomForestRegressor(), ['predict'], create_regression_problem_1()),
#             (CalibratedClassifierCV(LogisticRegression(), 'isotonic'), ['predict_proba'], create_weird_classification_problem_1()),
            (AdaBoostRegressor(), ['predict'], create_regression_problem_1()),
            (BaggingRegressor(), ['predict'], create_regression_problem_1()),
            (BaggingClassifier(), ['predict_proba'], create_weird_classification_problem_1()),
#             (GradientBoostingRegressor(verbose=True), ['predict'], create_regression_problem_1(m=100000, n=200)),
            (XGBRegressor(), ['predict'], create_regression_problem_for_xgb_1())
              ]
 
 
# Create tests for numpy renderer
def create_case_numpy(estimator, methods, fit_data, predict_data, export_predict_data):
    def test_case(self):
        model = clone(estimator)
        model.fit(**fit_data)
        for method in  methods:
            pred = getattr(model, method)(**predict_data)
            code = sklearn2code(model, method, numpy_renderer)
            try:
                module = exec_module('test_module', code)
                exported_pred = getattr(module, method)(export_predict_data['X'])
                if isinstance(exported_pred, tuple):
                    exported_pred = DataFrame(dict(enumerate(exported_pred)))
                assert_array_almost_equal(pred, exported_pred, 3)
            except:
#                 print(code)
                import clipboard
                clipboard.copy(code)
                raise
    test_case.__doc__ = ('Testing numpy language exportability of method%s %s of %s' % 
                         ('s' if len(methods)>1 else '', ', '.join(methods), repr(estimator)))
    return test_case
 
# All tests will be methods of this class
class TestExampleEstimatorsNumpy(object):
    pass

# The following loop adds a method to TestExampleEstimators for each test case
for i, (estimator, methods, (fit_data, predict_data, export_predict_data)) in enumerate(test_cases):
    case = create_case_numpy(estimator, methods, fit_data, predict_data, export_predict_data)
    case_name = 'test_case_%d' % i
    case.__name__ = case_name
    if PY2:
        case = MethodType(case, None, TestExampleEstimatorsNumpy)
    setattr(TestExampleEstimatorsNumpy, case_name, case)
    del case
#  
# # Create tests for numpy_flat language
# def create_case_numpy_flat(estimator, methods, fit_data, predict_data, export_predict_data):
#     def test_case(self):
#         model = clone(estimator)
#         model.fit(**fit_data)
#         for method in  methods:
#             pred = getattr(model, method)(**predict_data)
#             code = sklearn2code(model, method, numpy_flat)
#             try:
#                 module = exec_module('test_module', code)
#                 exported_pred = getattr(module, method)(**export_predict_data['X'])
#                 if isinstance(exported_pred, tuple):
#                     exported_pred = DataFrame(dict(enumerate(exported_pred)))
#                 assert_array_almost_equal(pred, exported_pred, 3)
#             except:
# #                 print(code)
# #                 import clipboard
# #                 clipboard.copy(code)
#                 raise
#     test_case.__doc__ = ('Testing numpy_flat language exportability of method%s %s of %s' % 
#                          ('s' if len(methods)>1 else '', ', '.join(methods), repr(estimator)))
#     return test_case
#  
# # All tests will be methods of this class
# class TestExampleEstimatorsNumpyFlat(object):
#     pass
#  
# # The following loop adds a method to TestExampleEstimators for each test case
# for i, (estimator, methods, (fit_data, predict_data, export_predict_data)) in enumerate(test_cases):
#     case = create_case_numpy_flat(estimator, methods, fit_data, predict_data, export_predict_data)
#     case_name = 'test_case_%d' % i
#     case.__name__ = case_name
#     if PY2:
#         case = MethodType(case, None, TestExampleEstimatorsNumpyFlat)
#     setattr(TestExampleEstimatorsNumpyFlat, case_name, case)
#     del case
#      
# # Create tests for pandas language
# def create_case_pandas(estimator, methods, fit_data, predict_data, export_predict_data):
#     def test_case(self):
#         model = clone(estimator)
#         model.fit(**fit_data)
#              
#         for method in  methods:
#             pred = DataFrame(getattr(model, method)(**predict_data))
#             try:
#                 code = sklearn2code(model, method, pandas)
#             except ExpressionTypeNotSupportedError:
#                 continue
#             try:
#                 module = exec_module('test_module', code)
#                 exported_pred = getattr(module, method)(export_predict_data['X'])
#                 assert_array_almost_equal(pred, exported_pred, 3)
#             except:
# #                 print(code)
#                 import clipboard
#                 clipboard.copy(code)
#                 raise
#     test_case.__doc__ = ('Testing pandas language exportability of method%s %s of %s' % 
#                          ('s' if len(methods)>1 else '', ', '.join(methods), repr(estimator)))
#     return test_case
#      
# # All tests will be methods of this class
# class TestExampleEstimatorsPandas(object):
#     pass
#      
# # The following loop adds a method to TestExampleEstimators for each test case
# for i, (estimator, methods, (fit_data, predict_data, export_predict_data)) in enumerate(test_cases):
#     case = create_case_pandas(estimator, methods, fit_data, predict_data, export_predict_data)
#     case_name = 'test_case_%d' % i
#     case.__name__ = case_name
#     if PY2:
#         case = MethodType(case, None, TestExampleEstimatorsPandas)
#     setattr(TestExampleEstimatorsPandas, case_name, case)
#     del case
#         
# def create_case_javascript(estimator, methods, fit_data, predict_data, export_predict_data):
#     def test_case(self):
#         model = clone(estimator)
#         model.fit(**fit_data)
#         for method in  methods:
#             try:
#                 code = sklearn2code(model, method, javascript)
#             except ExpressionTypeNotSupportedError:
#                 continue
#             js = execjs.get('Node')
#             context = js.compile(code)
#             exported_pred = []
#             
#             idx = np.random.binomial(1, 10. / float(export_predict_data['X'].shape[0]), size=export_predict_data['X'].shape[0])
#             for _, row in export_predict_data['X'].loc[idx, :].iterrows():
#                 val = context.eval('%s(%s)' % (method, ', '.join([str(x) if x==x else 'NaN' for x in row])))
#                 exported_pred.append(val)
#             exported_pred = np.array(exported_pred)
#             pred = DataFrame(getattr(model, method)(**predict_data))
#             assert_array_almost_equal(np.ravel(pred.loc[idx, :]), np.ravel(exported_pred), 3)
#     test_case.__doc__ = ('Testing javascript language exportability of method%s %s of %s' % 
#                          ('s' if len(methods)>1 else '', ', '.join(methods), repr(estimator)))
#     return test_case
#     
# # All tests will be methods of this class
# class TestExampleEstimatorsJavascript(object):
#     pass
#      
# # The following loop adds a method to TestExampleEstimators for each test case
# for i, (estimator, methods, (fit_data, predict_data, export_predict_data)) in enumerate(test_cases):
#     case = create_case_javascript(estimator, methods, fit_data, predict_data, export_predict_data)
#     case_name = 'test_case_%d' % i
#     case.__name__ = case_name
#     if PY2:
#         case = MethodType(case, None, TestExampleEstimatorsJavascript)
#     setattr(TestExampleEstimatorsJavascript, case_name, case)
#     del case

if __name__ == '__main__':
    # This code will run the test in this file.'
    import sys
    import nose
    module_name = sys.modules[__name__].__file__

    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s', '-v', '-x'])

    
    