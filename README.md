# Sklearn2Code
#### A flexible, extensible framework for converting scikit-learn models into portable software for deployment ####

Sklearn2Code converts most scikit-learn estimators into source code in different languages. 


### An Example 
Here's an example in which an `Earth` regressor is fitted to the Boston housing data set, then converted to Python code.
```python
from sklearn.datasets.base import load_boston
from sklearn.ensemble import RandomForestRegressor
from pandas import DataFrame
from sklearn2code.sklearn2code import sklearn2code
from sklearn2code.languages import numpy_flat
from sklearn2code.utility import exec_module
from numpy.testing.utils import assert_array_almost_equal
from yapf.yapflib.yapf_api import FormatCode
from sklearn.ensemble.forest import RandomForestRegressor

# Load a data set.
boston = load_boston()
X = DataFrame(boston['data'], columns=boston['feature_names'])
y = boston['target']

# Fit a scikit-learn model.
model = RandomForestRegressor().fit(X, y)

# Generate code from the scikit-learn model.
code = sklearn2code(model, ['predict'], numpy_flat)

# Write the code to a file
code_file = open("code_file.py","w")
code_file.write(code)
code_file.close()

#import the generated code
import code_file

# Confirm that the generated module produces output identical
# to the fitted model's predict method.
assert_array_almost_equal(model.predict(X), 
                          code_file.predict(**X))

# Print the generated code (using yapf for formatting).
print(FormatCode(code, style_config='pep8')[0])

```

### Installation

``` bash
$ pip install git+https://github.com/modusdatascience/sklearn2code
```
### Supported Estimators 

<table class="tg">

  <tr>
    <th class="tg-yw4l"></th>
    <th class="tg-yw4l">Python (Pandas)</th>
    <th class="tg-yw4l">Python (Numpy)</th>
    <th class="tg-yw4l">Javascript</th>
  </tr>
   <tr>
    <td class="tg-yw4l"><a href='http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html'>AdaBoost</a></td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
  </tr>
  <tr>
    <td class="tg-yw4l"><a href='http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html'>VotingClassifier</a></td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
  </tr>
  <tr>
    <td class="tg-yw4l"><a href='http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html'>GradientBoostingClassifier</a></td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
  </tr>
  <tr>
    <td class="tg-yw4l"><a href='http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html'>LogisticRegression</a></td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
  </tr>
  <tr>
    <td class="tg-yw4l"><a href='http://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html#sklearn.isotonic.IsotonicRegression'>IsotonicRegression</a></td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
  </tr>
  <tr>
    <td class="tg-yw4l"><a href='https://github.com/scikit-learn-contrib/py-earth'>PyEarth</a></td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
  </tr>
  <tr>
    <td class="tg-yw4l"><a href='http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html'>ElasticNet</a></td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
  </tr>
  <tr>
    <td class="tg-yw4l"><a href='http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html'>ElasticNetCV</a></td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
  </tr>
  <tr>
    <td class="tg-yw4l"><a href='http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html'>Lasso</a></td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
  </tr>
  <tr>
    <td class="tg-yw4l"><a href='http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html'>LassoCV</a></td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
  </tr>
  <tr>
    <td class="tg-yw4l"><a href='http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html'>Ridge</a></td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
  </tr>
  <tr>
    <td class="tg-yw4l"><a href='http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html'>RidgeCV</a></td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
  </tr>
  <tr>
    <td class="tg-yw4l"><a href='http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html'>SGDRegressor</a></td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
  </tr>
  <tr>
    <td class="tg-yw4l"><a href='http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html'>Pipeline</a></td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
  </tr>
  <tr>
    <td class="tg-yw4l"><a href='http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html'>FeatureUnion</a></td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
  </tr>
  <tr>
    <td class="tg-yw4l"><a href='http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html'>RandomForestRegressor</a></td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
  </tr>
  <tr>
    <td class="tg-yw4l"><a href='http://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html'>CalibratedClassifierCV</a></td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
    <td class="tg-yw4l">✓</td>
  </tr>
</table>

### An Example and Output 

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

### License ### 

Sklearn2Code is under the MIT license. 


