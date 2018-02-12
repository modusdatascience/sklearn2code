# A flexible, extensible framework for converting scikit-learn estimators into portable software

With `sklearn2code` you can generate source code in any language<sup>[1](#myfootnote1)</sup> from any scikit-learn estimator<sup>[2](#myfootnote2)</sup>.  
Here's an example in which an `Earth` regressor is fitted to the Boston housing data set, then converted to Python code.

```python
from sklearn.datasets.base import load_boston
from pyearth.earth import Earth
from pandas import DataFrame
from sklearn2code.sklearn2code import sklearn2code
from sklearn2code.languages import numpy_flat
from sklearn2code.utilty import exec_module
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
```

When run, the above program prints out the following code.

```python
from numpy import equal, where, isnan, maximum, minimum, exp, logical_not, logical_and, logical_or, select, less_equal, greater_equal, less, greater, nan, inf, log
from scipy.special import expit


def predict(CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B,
            LSTAT):
    return 31.0254749698609 + 0.30631323614756 * RAD + 18.6433070552184 * maximum(
        0, -6.431 + RM
    ) + 543.410344404762 * maximum(
        0, 1.4118 - DIS
    ) + 29.2561972599185 * maximum(0, 6.07 - LSTAT) - 2.0152457182328 * maximum(
        0, -1.4118 + DIS) + 2.47924143437217e-5 * B * maximum(
            0, 666.0 - TAX
        ) + 0.00135332955636613 * maximum(0, 371.72 - B) * maximum(
            0, -6.07 + LSTAT) + 0.00544318002937416 * PTRATIO * maximum(
                0, 74.3 - AGE
            ) + 0.00594283440699428 * maximum(0, -6.07 + LSTAT) * maximum(
                0, -371.72 + B
            ) + 0.0435936808733572 * maximum(0, 7.99248 - CRIM) * maximum(
                0, -6.07 + LSTAT) + 0.180064471129214 * RAD * maximum(
                    0, 6.861 - RM
                ) + 0.186888125469055 * maximum(0, -2.5975 + DIS) * maximum(
                    0, -1.4118 + DIS
                ) + 0.212787281495065 * maximum(0, 6.431 - RM) * maximum(
                    0, -9.08 + LSTAT) + 0.505186705211807 * PTRATIO * maximum(
                        0, 4.906 - RM) - 0.00106563539861781 * RAD * maximum(
                            0, 378.35 - B
                        ) - 0.0100576495224232 * RAD * maximum(
                            0, -378.35 + B
                        ) - 0.00508038092469576 * PTRATIO * maximum(
                            0, -4.22239 + CRIM
                        ) - 0.324766921858327 * PTRATIO * maximum(
                            0, -4.906 + RM) - 0.070793219135922 * B * maximum(
                                0, 6.07 - LSTAT
                            ) - 0.746293835428375 * RAD * maximum(
                                0, -6.861 + RM
                            ) - 0.613111567305932 * PTRATIO * maximum(
                                0, 6.383 - RM
                            ) - 0.000423644678943647 * TAX * maximum(
                                0,
                                -6.07 + LSTAT) - 0.0144177152492291 * maximum(
                                    0, 56.7 - AGE) * maximum(
                                        0, -1.4118 + DIS
                                    ) - 5.35368370525657 * maximum(
                                        0, 2.5975 - DIS) * maximum(
                                            0, -1.4118 + DIS
                                        ) - 1.00102238059424 * NOX * maximum(
                                            0, -6.07 + LSTAT
                                        ) - 7.806533030774 * CRIM * maximum(
                                            0, 1.4118 - DIS
                                        ) - 613.297531274621 * NOX * maximum(
                                            0, 1.4118 - DIS)

```

<a name="myfootnote1">1</a>: In some cases you have to implement support for your desired language.  We try to make this as easy as possible.
<a name="myfootnote2">2</a>: `sklearn2code` also supports some external scikit-learn compatible packages, such as [py-earth](https://github.com/scikit-learn-contrib/py-earth), and is designed to be easily extended to support other types of estimators.  


