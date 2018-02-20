from sklearn.datasets.base import load_boston
from pyearth.earth import Earth
from pandas import DataFrame
from sklearn2code.sklearn2code import sklearn2code
from sklearn2code.languages import numpy_flat
from sklearn2code.utility import exec_module
from numpy.testing.utils import assert_array_almost_equal
from yapf.yapflib.yapf_api import FormatCode

# Load a data set.
boston = load_boston()
X = DataFrame(boston['data'], columns=boston['feature_names'])
y = boston['target']

# Fit a py-earth model.
model = Earth(max_degree=2).fit(X, y)

# Generate code from the py-earth model.
code = sklearn2code(model, ['predict'], numpy_flat)

# Execute the generated code in its own module.
boston_housing_module = exec_module('boston_housing_module', code)

# Confirm that the generated module produces output identical
# to the fitted model's predict method.
assert_array_almost_equal(model.predict(X), 
                          boston_housing_module.predict(**X))

# Print the generated code (using yapf for formatting).
print(FormatCode(code, style_config='pep8')[0])