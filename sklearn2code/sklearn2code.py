from .sym.function import tupify
from sklearn2code.utility import import_submodules
from .sym import adapters

# Register all the adapters
import_submodules(adapters, ignore_import_errors=False)

def sklearn2code(estimator, methods, renderer, trim=True, argument_names=None, **extra_args):
    return renderer.generate(estimator, tupify(methods), trim=trim, argument_names=argument_names, **extra_args)


