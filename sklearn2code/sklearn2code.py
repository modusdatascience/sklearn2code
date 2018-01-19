from .sym.base import sym_predict, sym_predict_proba, sym_transform


def sklearn2code(estimator, methods, language, trim=True, **extra_args):
    return language.generate(estimator, methods, trim=trim, **extra_args)

