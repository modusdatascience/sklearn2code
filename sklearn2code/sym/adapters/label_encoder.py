from ..base import sym_transform
from sklearn2code.sym.expression import as_value, StringVariable, FiniteMap
from sklearn2code.sym.function import Function
from sklearn2code.sym.base import sym_inverse_transform
from sklearn.preprocessing.label import LabelEncoder
from toolz.functoolz import compose
import numpy as np
from toolz.curried import keymap

def np_to_py(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj

@sym_transform.register(LabelEncoder)
def sym_transform_label_encoder(estimator):
    mapping = dict(map(reversed, enumerate(map(as_value, estimator.classes_))))
    arg = StringVariable('x')
    return Function.from_expression(FiniteMap(mapping=mapping, arg = arg))

@sym_inverse_transform.register(LabelEncoder)
def sym_inverse_transform_label_encoder(estimator):
    mapping = keymap(as_value, dict(enumerate(map(compose(as_value, np_to_py), estimator.classes_))))
    arg = StringVariable('x')
    return Function.from_expression(FiniteMap(mapping=mapping, arg=arg))