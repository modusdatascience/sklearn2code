from six import with_metaclass
from abc import ABCMeta, abstractproperty, abstractmethod


class Renderer(with_metaclass(ABCMeta, object)):
    @abstractmethod
    def render(self, functions, **extra_args):
        '''
        
        Parameters
        ----------
        
        functions (iterable of Functions): The functions to render, sorted in topological order so that all 
            functions are defined before they are called by other functions.
        
        '''


class StructuredRenderer(Renderer):
    def render(self, functions, **extra_args):
        pass
    
    @abstractmethod
    def render_assignment(self, assigment):
        pass
    
    @abstractmethod
    def render_splat_assignment(self, assignment):
        pass
    
    @abstractmethod
    def assigment(self, lhs, rhs):
        pass
    
    @abstractmethod
    