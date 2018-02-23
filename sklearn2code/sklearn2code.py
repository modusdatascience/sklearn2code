from .sym.function import tupify

def sklearn2code(estimator, methods, language, trim=True, **extra_args):
    return language.generate(estimator, tupify(methods), trim=trim, **extra_args)

from .sym import adapters
import importlib
import pkgutil
import traceback

# https://stackoverflow.com/a/25562415/1572508
def import_submodules(package, recursive=True, ignore_import_errors=True):
    """ Import all submodules of a module, recursively, including subpackages

    :param package: package (name or actual module)
    :type package: str | module
    :rtype: dict[str, types.ModuleType]
    """
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + '.' + name
        try:
            results[full_name] = importlib.import_module(full_name)
        except ImportError:
            if not ignore_import_errors:
                raise
            traceback.print_exc()
        if recursive and is_pkg:
            results.update(import_submodules(full_name))
    return results

import_submodules(adapters, ignore_import_errors=False)
