import imp
from six import exec_, string_types
import numpy as np
from toolz.functoolz import curry
from operator import __add__
from six.moves import reduce
from multipledispatch.dispatcher import Dispatcher
import importlib
import pkgutil

# https://stackoverflow.com/a/25562415/1572508
def import_submodules(package, recursive=True, ignore_import_errors=True):
    """ 
    Import all submodules of a module, recursively, including subpackages.
    
    Parameters
    ----------
    
    package (str or types.ModuleType): Package name or actual module.
    
    recursive (bool): Whether or not to recursively import from subpackages.
    
    ignore_import_errors (bool): Whether or not to ignor ImportErrors in imported modules.  Set to 
        True if importing modules that are optional and rely on optional dependencies.
        
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
        if recursive and is_pkg:
            results.update(import_submodules(full_name))
    return results

class InnerDispatcher(object):
    def __init__(self, parent, instance):
        self.parent = parent
        self.instance = instance
        
    def __call__(self, *args):
        return self.parent.dispatcher(self.instance, *args)
    
    def register(self, *types):
        return self.parent.register(*types)

class HeritableDispatcher(object):
    def __init__(self, name):
        self.dispatcher = Dispatcher(name)
        
    def register(self, *types):
        def _register(fun):
            fun._dispatcher_1f0irrij9 = self
            fun._dispatch_types_2m20fi4rin = tuple(types)
            return fun
        return _register
    
    def __get__(self, instance, owner):
        return InnerDispatcher(self, instance)

class HeritableDispatcherMeta(type):
    def __init__(self, name, bases, dct):
        for method in dct.values():
            if hasattr(method, '_dispatch_types_2m20fi4rin'):
                method._dispatcher_1f0irrij9.dispatcher.register(self, *method._dispatch_types_2m20fi4rin)(method)
        return type.__init__(self, name, bases, dct)


@curry
def tupsmap(n, fun, tups):
    return tuple([tup[:n] + (fun(tup[n]),) + tup[(n+1):] for tup in tups])

@curry
def tupapply(tup):
    return tup[0](*tup[1:])

def tupfun(*funs):
    def _tupfun(tup):
        return tuple(map(tupapply, zip(funs, tup)))
    return _tupfun

@curry
def tupget(n, tup):
    return tup[n]

def isiterable(obj):
    return hasattr(obj, '__iter__')

def tupshape(obj):
    if isiterable(obj) and not isinstance(obj, string_types):
        return tupshape(obj)
    else:
        return tuple()

@curry
def tupall(predicate, obj):
    if isiterable(obj) and not isinstance(obj, string_types):
        return all(map(tupall(predicate), obj))
    else:
        return predicate(obj)

@curry
def tupany(predicate, obj):
    if isiterable(obj) and not isinstance(obj, string_types):
        return any(map(tupall(predicate), obj))
    else:
        return predicate(obj)

def flatten(obj):
    if isiterable(obj) and not isinstance(obj, string_types):
        return reduce(__add__, map(flatten, obj), tuple())
    else:
        return (obj,)

def tupify(obj):
    if isiterable(obj) and not isinstance(obj, string_types):
        return tuple(obj)
    else:
        return (obj,)

def exec_module(name, code):
    module = imp.new_module(name)
    exec_(code, module.__dict__)
    return module

def xlabels(X):
    try:
        labels = list(X.columns)
    except AttributeError:
        try:
            labels = list(X.design_info.column_names)
        except AttributeError:
            try:
                labels = list(X.dtype.names)
            except TypeError:
                try:
                    labels = ['x%d' % i for i in range(X.shape[1])]
                except IndexError:
                    labels = ['x%d' % i for i in range(1)]
            # handle case where X is not np.array (e.g list)
            except AttributeError:
                X = np.array(X)
                labels = ['x%d' % i for i in range(X.shape[1])]
    return labels

    