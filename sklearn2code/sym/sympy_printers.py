from toolz.functoolz import curry
from sympy.printing.jscode import JavascriptCodePrinter
from sympy.printing.lambdarepr import NumPyPrinter
from sympy.printing.python import PythonPrinter

@curry
def reduction(function_name, self, args):
    if len(args) == 1:
        return self._print(args[0])
    else:
        return function_name + '(' + self._print(args[0]) + ',' + reduction(function_name, self, args[1:]) + ')'


class S2CJavaScriptPrinter(JavascriptCodePrinter):
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


class S2CNumpyPrinter(NumPyPrinter):
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

class S2CPythonPrinter(PythonPrinter):
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

