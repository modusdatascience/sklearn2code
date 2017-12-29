from abc import abstractmethod, ABCMeta

class Function(object):
    def __init__(self, inputs, calls, output, origin=None):
        '''
        Parameters
        ----------
        inputs : tuple of sympy Symbols
            The input variables for this function.
        
        calls : dict with keys tuples of Symbols and values pairs of Function objects and dicts of their inputs,
        where the dicts of inputs have keys and values that are symbols.
            The values are other function calls made by this function.  The keys are 
            variables to which the outputs are assigned.  The number of output symbols in the
            key must match the number of outputs in the Function.  The length of the tuple of inputs must match the
            number of inputs for the function.  Also, no two keys may contain 
            the same variable symbol.  These constraints are checked with assertions.
            
        output : tuple of sympy expressions
            The actual calculations made by this Function.  The return values of theFfunction
            are the results of the computations expressed by the expressions.
        
        name : object or None
            The origin of this function, usually a scikit-learn estimator.  This may be used by a Serializer or 
            NamingScheme if available.
        '''
        self.inputs = inputs
        self.calls = calls
        self.output = output
        self.origin = origin
        self._validate()
    
    def _validate(self):
        sym_set = set()
        for syms, (function, inputs) in self.calls:
            for sym in syms:
                assert sym not in sym_set
                sym_set.add(sym)
            assert len(syms) == len(function.output)
            assert len(inputs) == len(function.inputs)

class NamingSchemeBase(object):
    __metaclass__ = ABCMeta
    def name(self, function):
        '''
        Assign names to Function and any Functions called by Function.
        
        Parameters
        ----------
        function : instance of Function
        
        Returns
        -------
        dict with keys Functions and values strs
        '''

class SerializerBase(object):
    __metaclass__ = ABCMeta
    def serialize(self, functions):
        '''
        Serialize the function.
        
        Parameters
        ----------
        functions : dict with keys strs and values instances of Function.  The keys are typically used to 
        name the functions.
        
        Returns
        -------
        str
            A string representing the serialized Functions.  Usually this should be written to a file.
        '''
        return self._serialize(functions)
    
    @abstractmethod
    def _serialize(self, function):
        pass
    
class PrinterTemplateSerializer(SerializerBase):
    def __init__(self, printer, template):
        '''
        Parameters
        ----------
        printer : An instance of a CodePrinter subclass from sympy.
        template : A mako template.
        
        '''
        self.printer = printer
        self.template = template

            
            
        
    
    

