import imp
from six import exec_
import numpy as np

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

    