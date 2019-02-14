import os
from .numpy import renderer as numpy
from sklearn2code.renderers.base import BasicRenderer
from sklearn2code.sym.printers import NumpyPrinter, JavascriptPrinter,\
    PandasPrinter
from mako.template import Template
template_dir = os.path.dirname(os.path.abspath(__file__))

numpy_flat = BasicRenderer(NumpyPrinter(), Template(filename=os.path.join(template_dir, 'numpy_flat_template.mako.py')))
numpy_flat_kwargs = BasicRenderer(NumpyPrinter(), Template(filename=os.path.join(template_dir, 'numpy_flat_kwargs_template.mako.py')))
pandas = BasicRenderer(PandasPrinter(), Template(filename=os.path.join(template_dir, 'pandas_template.mako.py')))
javascript = BasicRenderer(JavascriptPrinter(), Template(filename=os.path.join(template_dir, 'javascript_template.mako.js')))

