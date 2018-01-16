from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.base import clone
from toolz.dicttoolz import merge
from pandas.core.frame import DataFrame
import numpy as np
from sklearn2code.sym.base import sym_predict_proba

def test_gradient_boosting_classifier():
    m = 1000
    n = 10
    X = DataFrame(np.random.normal(size=(m,n)), columns=['x%d' % i for i in range(n)])
    thresh = np.random.normal(size=n)
    X_transformed = X * (X > thresh)
    beta = np.random.normal(size=n)
    y = (np.dot(X_transformed, beta) + np.random.normal(size=m)) > 0
    
#     methods = ['predict_proba']
    estimator = GradientBoostingClassifier(max_depth=10, n_estimators=10)
    model = clone(estimator)
    model.fit(X, y)
    fun = sym_predict_proba(model)
    print(fun)
#     
#     for method in  methods:
#         pred = getattr(model, method)(**predictor_data)
#         if len(pred.shape) > 1:
#             pred = pred[:,1]
#         code = model_to_code(model, 'numpy', method, 'test_model')      
#         module = exec_module('test_module', code)
#         exported_pred = getattr(module, 'test_model')(**predictor_data['X'])
#         assert_array_almost_equal(pred, exported_pred)
if __name__ == '__main__':
    # This code will run the test in this file.'
    import sys
    import nose
    module_name = sys.modules[__name__].__file__

    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s', '-v'])


