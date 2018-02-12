from .sym.function import tupify


def sklearn2code(estimator, methods, language, trim=True, **extra_args):
    return language.generate(estimator, tupify(methods), trim=trim, **extra_args)

