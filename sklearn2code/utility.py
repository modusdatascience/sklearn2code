import imp
from six import exec_, string_types
import numpy as np
from toolz.functoolz import curry
from operator import __add__
from six.moves import reduce

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

    