from .sym.base import sym_predict, sym_predict_proba, sym_transform
from .sym.function import tupify


def sklearn2code(estimator, methods, language, trim=True, **extra_args):
    return language.generate(estimator, tupify(methods), trim=trim, **extra_args)

