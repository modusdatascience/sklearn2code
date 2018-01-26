from .sym.base import sym_predict, sym_predict_proba, sym_transform
from .sym.function import tupapply
from itertools import repeat
import networkx
from six.moves import reduce
from operator import methodcaller, __mod__
from toolz.functoolz import curry, complement, compose
from toolz.dicttoolz import merge
from .sym.function import tupsmap
from .sym.sympy_printers import S2CNumpyPrinter
from mako.template import Template
import os
from sklearn2code.templates import template_dir
from networkx.classes.digraph import DiGraph

method_dispatcher = dict(
                         predict = sym_predict,
                         predict_proba = sym_predict_proba,
                         transform = sym_transform
                         )

class Language(object):
    def __init__(self, printer, template):
        '''
        Parameters
        ----------
        
        printer : callable
            A callable that takes a sympy expression and returns a string.
            
        template : Template
            A mako Template that takes an iterable of Functions, a printer, 
            a namer, and optional extra arguments.
        '''
        self.printer = printer
        self.template = template
    
    def generate(self, estimator, methods, trim, **extra_args):
        functions = tuple(map(tupapply, zip(map(method_dispatcher.__getitem__, methods), repeat(estimator))))
        g = reduce(networkx.compose, map(methodcaller('digraph'), functions), DiGraph())
        sorted_functions = tuple(networkx.topological_sort(g))
        names = dict(zip(functions, methods))
        unnamed = tuple(filter(complement(names.__contains__), sorted_functions))
        names = merge(names, dict(tupsmap(1, curry(__mod__)('_f%d'), map(compose(tuple,reversed), enumerate(unnamed)))))
        return self.template.render(functions=sorted_functions, printer=self.printer, namer=names.__getitem__, **extra_args)
    
numpy_flat = Language(S2CNumpyPrinter().doprint, Template(filename=os.path.join(template_dir, 'numpy_flat_template.mako.py')))



