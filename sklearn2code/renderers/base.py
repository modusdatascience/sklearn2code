from six import with_metaclass
from abc import ABCMeta, abstractmethod
from sklearn2code.sym.function import VariableNameFactory, Function, safe_symbol,\
    toposort
from itertools import repeat
from sklearn2code.utility import tupapply, tupsmap
from toolz.functoolz import complement, curry, compose
from operator import __mod__
from toolz.dicttoolz import merge
from sklearn2code.sym.base import sym_predict, sym_predict_proba, sym_transform

method_dispatcher = dict(
                         predict = sym_predict,
                         predict_proba = sym_predict_proba,
                         transform = sym_transform
                         )

class Renderer(with_metaclass(ABCMeta, object)):
    @abstractmethod
    def render(self, functions, namer, **extra_args):
        '''
        
        Parameters
        ----------
        
        functions (iterable of Functions): The functions to render, sorted in topological order so that all 
            functions are defined before they are called by other functions.
        
        '''
    
    def generate(self, estimator, methods, trim, argument_names, **extra_args):
        functions = tuple(map(tupapply, zip(map(method_dispatcher.__getitem__, methods), repeat(estimator))))
        if argument_names is not None:
            outer = Function(tuple(map(safe_symbol, argument_names)), tuple(), tuple(map(safe_symbol, argument_names)))
            functions = tuple(map(lambda x: x.compose(outer), functions))
        sorted_functions = toposort(functions)
        names = dict(zip(functions, methods))
        unnamed = tuple(filter(complement(names.__contains__), sorted_functions))
        names = merge(names, dict(tupsmap(1, curry(__mod__)('_f%d'), map(compose(tuple,reversed), enumerate(unnamed)))))
        return self.render(functions=sorted_functions,
                                    namer=names.__getitem__, **extra_args)

class BasicRenderer(Renderer):
    def __init__(self, printer, template):
        self.printer = printer
        self.template = template
        
    def render(self, functions, namer, **extra_args):
        return self.template.render(functions=functions, printer=self.printer, 
                                    namer=namer, **extra_args)

class StructuredRenderer(Renderer):
    def __init__(self, printer, file_template, function_template, 
                 assignment_template, unpacking_assignment_template):
        self.printer = printer
        self.file_template = file_template
        self.function_template = function_template
        self.assignment_template = assignment_template
        self.unpacking_assignment_template = unpacking_assignment_template

    def render(self, functions, namer, **extra_args):
        rendered_functions = []
        for function in functions:
            rendered_functions.append(self.render_function(function, namer))
        result = self.render_file(rendered_functions)
        return result
    
    def render_assignment(self, lhs, called_function_name, inputs, arguments, Var):
        return self.assignment_template.render(renderer=self, lhs=lhs, called_function_name=called_function_name, 
                                        inputs=inputs, arguments=arguments, Var=Var)
    
    def render_unpacking_assignment(self, lhs, called_function_name, inputs, arguments, Var):
        return self.assignment_template.render(renderer=self, lhs=lhs, called_function_name=called_function_name, 
                                        inputs=inputs, arguments=arguments, Var=Var)
    
    def render_function(self, function, namer):
        Var = VariableNameFactory(existing=function.all_variables())
        rendered_assignments = []
        for lhs, (called_function, arguments) in function.calls:
            called_function_name = namer(called_function)
            if len(lhs) > len(called_function.outputs):
                # Since the Function has already checked, we 
                # know that this assignment is unpackable.
                rendered_assignments.append(self.render_unpacking_assignment(lhs=lhs, 
                                                                             called_function_name=called_function_name, 
                                                                             inputs=called_function.inputs, 
                                                                             arguments=arguments, 
                                                                             Var=Var))
            else:
                rendered_assignments.append(self.render_assignment(lhs=lhs, 
                                                                   called_function_name=called_function_name, 
                                                                   inputs=called_function.inputs, 
                                                                   arguments=arguments, 
                                                                   Var=Var))
        function_name = namer(function)
        return self.function_template.render(renderer=self, function_name=function_name,
                                             inputs=function.inputs,
                                             rendered_assignments=rendered_assignments,
                                             outputs=function.outputs)
        
        
        
    def render_file(self, rendered_functions):
        return self.file_template.render(rendered_functions=rendered_functions)

