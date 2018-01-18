from sympy.printing.jscode import JavascriptCodePrinter
import os
from sklearn2code.sym.resources import resources
from mako.template import Template
from sympy.printing.lambdarepr import NumPyPrinter
from sympy.printing.python import PythonPrinter
from _collections import defaultdict
from operator import add
import imp
from six import exec_
from toolz.functoolz import curry

def exec_module(name, code):
    module = imp.new_module(name)
    exec_(code, module.__dict__)
    return module

@curry
def reduction(function_name, self, args):
    if len(args) == 1:
        return self._print(args[0])
    else:
        return function_name + '(' + self._print(args[0]) + ',' + reduction(function_name, self, args[1:]) + ')'


class STJavaScriptPrinter(JavascriptCodePrinter):
    def _print_Max(self, expr):
        return 'Math.max(' + ','.join(self._print(i) for i in expr.args) + ')'
    
    def _print_Min(self, expr):
        return 'Math.min(' + ','.join(self._print(i) for i in expr.args) + ')'
 
    def _print_NaNProtect(self, expr):
        return 'nanprotect(' + ','.join(self._print(i) for i in expr.args) + ')'
 
    def _print_Missing(self, expr):
        return 'missing(' + ','.join(self._print(a) for a in expr.args) + ')'
    
    def _print_Expit(self, expr):
        return 'expit(' + ','.join(self._print(i) for i in expr.args) + ')'
    
    def _print_NAN(self, expr):
        return 'NaN'
    
javascript_template_filename = os.path.join(resources, 'javascript_template.mako.js')
with open(javascript_template_filename) as infile:
    javascript_template = Template(infile.read())
javascript_function_template_filename = os.path.join(resources, 'javascript_function_template.mako.js')
with open(javascript_function_template_filename) as infile:
    javascript_function_template = Template(infile.read())

def javascript_assigner(symbols, function_name, input_symbols):
    return 'var [%s] = %s(%s)' % (', '.join(symbols), function_name, ','.join(input_symbols))
    
# def javascript_str(function_name, estimator, method=sym_predict, all_variables=False):
#     expression = method(estimator)
#     used_names = expression.free_symbols
#     input_names = [sym.name for sym in syms(estimator) if sym in used_names or all_variables]
#     return javascript_template.render(function_name=function_name, input_names=input_names,
#                                       function_code=STJavaScriptPrinter().doprint(expression))


class STNumpyPrinter(NumPyPrinter):
    def _print_And(self, expr):
        return reduction('logical_and', self, expr.args)
    
    def _print_Or(self, expr):
        return reduction('logical_or', self, expr.args)
    
    def _print_Max(self, expr):
        return 'maximum(' + ','.join(self._print(i) for i in expr.args) + ')'
    
    def _print_Min(self, expr):
        return 'minimum(' + ','.join(self._print(i) for i in expr.args) + ')'

    def _print_NaNProtect(self, expr):
        return 'where(isnan(' + ','.join(self._print(a) for a in expr.args) + '), 0, ' \
            + ','.join(self._print(a) for a in expr.args) + ')'

    def _print_Missing(self, expr):
        return 'isnan(' + ','.join(self._print(a) for a in expr.args) + ').astype(float)'
    
    def _print_Expit(self, expr):
        return 'expit(' + ','.join(self._print(i) for i in expr.args) + ')'
    
    def _print_NAN(self, expr):
        return 'nan'
    
numpy_template_filename = os.path.join(resources, 'numpy_template.mako.py')
with open(numpy_template_filename) as infile:
    numpy_template = Template(infile.read())
numpy_function_template_filename = os.path.join(resources, 'numpy_function_template.mako.py')
with open(numpy_function_template_filename) as infile:
    numpy_function_template = Template(infile.read())

# def numpy_str(function_name, estimator, method=sym_predict, all_variables=False, pep8=False):
#     expression = method(estimator)
#     used_names = expression.free_symbols
#     input_names = [sym.name for sym in syms(estimator) if sym in used_names or all_variables]
#     function_code = STNumpyPrinter().doprint(expression)
#     result = numpy_template.render(function_name=function_name, input_names=input_names,
#                                       function_code=function_code)
#     if pep8:
#         result =  autopep8.fix_code(result, options={'aggressive': 1})
#     return result

class STPythonPrinter(PythonPrinter):
    def _print_Float(self, expr):
        return str(expr)
    
    def _print_Not(self, expr):
        return 'negate(' + ','.join(self._print(i) for i in expr.args) + ')'
    
    def _print_Max(self, expr):
        return 'max(' + ','.join(self._print(i) for i in expr.args) + ')'
    
    def _print_Min(self, expr):
        return 'min(' + ','.join(self._print(i) for i in expr.args) + ')'
    
    def _print_NaNProtect(self, expr):
        return 'nanprotect(' + ','.join(self._print(i) for i in expr.args) + ')'

    def _print_Missing(self, expr):
        return 'missing(' + ','.join(self._print(i) for i in expr.args) + ')'
    
    def _print_Expit(self, expr):
        return 'expit(' + ','.join(self._print(i) for i in expr.args) + ')'
    
    def _print_NAN(self, expr):
        return 'float(\'nan\')'

python_template_filename = os.path.join(resources, 'python_template.mako.py')
with open(python_template_filename) as infile:
    python_template = Template(infile.read())
python_function_template_filename = os.path.join(resources, 'python_function_template.mako.py')
with open(python_function_template_filename) as infile:
    python_function_template = Template(infile.read())

# def python_str(function_name, estimator, method=sym_predict, all_variables=False):
#     expression = method(estimator)
#     used_names = expression.free_symbols
#     input_names = [sym.name for sym in syms(estimator) if sym in used_names or all_variables]
#     return autopep8.fix_code(python_template.render(function_name=function_name, input_names=input_names,
#                                       function_code=STPythonPrinter().doprint(expression)), options={'aggressive': 1})

language_print_dispatcher = {
    'python': STPythonPrinter,
    'numpy': STNumpyPrinter,
    'javascript': STJavaScriptPrinter
    }

language_template_dispatcher = {
    'python': python_template,
    'numpy': numpy_template,
    'javascript': javascript_template 
    }

language_function_template_dispatcher = {
    'python': python_function_template,
    'numpy': numpy_function_template,
    'javascript': javascript_function_template
    }

language_assignment_statement_dispatcher = defaultdict(lambda: lambda symbols, function_name, input_symbols: ', '.join(symbols) + ' = %s(%s)' % (function_name, ', '.join(input_symbols)) )
language_assignment_statement_dispatcher['javascript'] = javascript_assigner
language_assignment_statement_dispatcher['numpy'] = lambda symbols, function_name, input_symbols: ', '.join(symbols) + ' = %s(**kwargs)' % function_name
language_return_statement_dispatcher = defaultdict(lambda: lambda expressions: 'return ' + ', '.join(expressions))
language_return_statement_dispatcher['javascript'] = lambda expressions: 'return [' + ', '.join(expressions) + ']'
# 
# def trim_code_precursors(assignments, outputs, inputs, all_variables):
#     reverse_new_assignments = []
#     new_inputs = []
#     used = set(reduce(add, map(lambda x: x.free_symbols, outputs)))
#     for variable, expr in reversed(assignments):
#         if variable in used:
#             used.update(expr.free_symbols)
#             reverse_new_assignments.append((variable, expr))
#     if not all_variables:
#         for variable in inputs:
#             if variable in used:
#                 new_inputs.append(variable)
#     else:
#         new_inputs.extend(inputs)
#     return reversed(reverse_new_assignments), new_inputs
#             
            

# def assignment_pairs_and_outputs_to_code(pairs_and_outputs, language, function_name, inputs, all_variables):
#     assignments, outputs = pairs_and_outputs
#     assignment_statements = ''
#     assignments, inputs_ = trim_code_precursors(assignments, outputs, inputs, all_variables)
#     
#     printer = language_print_dispatcher[language]
#     assigner = language_assignment_statement_dispatcher[language]
#     returner = language_return_statement_dispatcher[language]
#     template = language_template_dispatcher[language]
#     for symbol, expression in assignments:
#         
#         assignment_statements += assigner(symbol.name, printer().doprint(expression)) + '\n'
#     
#     return_statement = returner(map(printer().doprint, outputs))
#     return template.render(function_name=function_name, input_names=map(lambda x: x.name, inputs_), 
#                            assignment_code=assignment_statements, return_code=return_statement)

def parts_to_code(parts, language, function_name, all_variables):
    
    function_template = language_function_template_dispatcher[language]
    printer = language_print_dispatcher[language]().doprint
    assigner = language_assignment_statement_dispatcher[language]
    returner = language_return_statement_dispatcher[language]
    template = language_template_dispatcher[language]
    
    if not all_variables:
        parts = trim_parts(parts)
    
    first_inputs, expressions, target = parts
    inputs = first_inputs
    index = 0
    functions = []
    previous = None
    while True:
        if previous is None:
            body_code = ''
        else:
            if previous[0]:
                body_code = assigner(*previous)
            else:
                body_code = ''
        if target is None:
            name = function_name
        else:
            name = '_' + function_name + '_' + str(index)
            index += 1
            
        return_code = returner(map(printer, expressions))
        function = function_template.render(function_name=name, input_names=list(map(lambda x: x.name, first_inputs)), 
                                            body_code=body_code, return_code=return_code)
        functions.append(function)
        if target is not None:
            target_inputs, _, _ = target
            previous = (list(map(lambda x: x.name, target_inputs)), name, list(map(lambda x: x.name, first_inputs)))
            inputs, expressions, target = target
        else:
            break
    result = template.render(functions = functions)
    return result
#     pairs_and_outputs = assemble_parts_into_assignment_pairs_and_outputs(parts)
#     inputs = [symbol for symbol in parts[0]]
#     return assignment_pairs_and_outputs_to_code(pairs_and_outputs, language, function_name, inputs, all_variables)

model_to_code_method_dispatch = {'predict': sym_predict_parts,
                                 'transform': sym_transform_parts,
                                 'predict_proba': sym_predict_proba_parts}

def model_to_code(model, language, method, function_name, all_variables=False):
    parts = model_to_code_method_dispatch[method](model)
    assert_parts_are_composable(parts)
    result = parts_to_code(parts, language, function_name, all_variables)
    return result
    
