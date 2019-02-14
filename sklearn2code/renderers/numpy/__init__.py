import os
from mako.template import Template
from sklearn2code.sym.printers import NumpyPrinter
from sklearn2code.renderers.base import StructuredRenderer

template_dir = os.path.dirname(__file__)
assignment_template = Template(filename=os.path.join(template_dir, 'assignment_template.mako.py'))
unpacking_assignment_template = Template(filename=os.path.join(template_dir, 'unpacking_assignment_template.mako.py'))
function_template = Template(filename=os.path.join(template_dir, 'function_template.mako.py'))
file_template = Template(filename=os.path.join(template_dir, 'file_template.mako.py'))

renderer = StructuredRenderer(printer=NumpyPrinter(), assignment_template=assignment_template,
                              unpacking_assignment_template=unpacking_assignment_template,
                              function_template=function_template,
                              file_template=file_template)

