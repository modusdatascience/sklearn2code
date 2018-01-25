from sklearn2code.sym.base import register_sym_transform, sym_transform
from toolz.functoolz import compose
from sklearn2code.sym.function import tupget, cart
from operator import __mul__
from itertools import starmap
from sklearn.pipeline import FeatureUnion

@register_sym_transform(FeatureUnion)
def sym_transform_feature_union(estimator):
    keys = tuple(map(tupget(0), estimator.transformer_list))
    transformers = map(compose(sym_transform, tupget(1)), estimator.transformer_list)
    weights = map(estimator.transformer_weights.__getitem__, keys)
    return cart(*starmap(__mul__, zip(weights, transformers)))

