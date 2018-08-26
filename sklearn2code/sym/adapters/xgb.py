from xgboost.sklearn import XGBRegressor
from sklearn.datasets.samples_generator import make_regression
from pandas.core.frame import DataFrame
from pyparsing import Literal, Regex, Optional, Word, alphas, OneOrMore, Forward,\
    indentedBlock, Suppress, Group
from sklearn2code.sym.expression import RealVariable, RealNumber, IsNan,\
    Piecewise, true
from sklearn2code.sym.base import sym_predict
from sklearn2code.sym.function import VariableFactory, Function
from operator import methodcaller, __add__
from toolz.functoolz import compose
from toolz.itertoolz import first, last
from six.moves import reduce
from sklearn2code.sklearn2code import sklearn2code
from sklearn2code.languages import numpy_flat
from sklearn2code.utility import exec_module

class Node(object):
    @classmethod
    def parser(self):
        integer = Regex('\d+').setParseAction(lambda x,y,z: int(z[0]))
        floating_point = Regex('[+-]?(\d*[.])?\d+(e[+-]\d+)?').setParseAction(lambda x,y,z: float(z[0]))#Group(Optional(Suppress('-')) + Regex('\d+') + Optional(Literal('.') + Regex('\d+')))
        left_bracket = Suppress('[')
        right_bracket = Suppress(']')
        variable_name = Word(alphas + '0123456789_')
        lt = Literal('<')
        branch = (integer + Suppress(':') + left_bracket +  variable_name + lt + 
                  floating_point + right_bracket + Group(Literal('yes=') + integer) + Suppress(',') + 
                  Group(Literal('no=') + integer) + Suppress(',') + Group(Literal('missing=') + integer))
        # branch
        leaf = integer + Suppress(':') + Suppress('leaf=') + floating_point
        
        stack = [1]
        tree = Forward()
        tree << (leaf ^ (branch + indentedBlock(tree, stack)))
        return OneOrMore(tree)
    
    def __init__(self, variable, threshold, yes, no, missing):
        self.variable = variable
        self.threshold = threshold
        self.yes = yes
        self.no = no
        self.missing = missing
    
    @classmethod
    def from_parsed(cls, parsed):
        if len(parsed) == 2:
            return parsed[-1]
        variable = parsed[1]
        threshold = parsed[3]
        subreferences = dict(parsed[4:7])
        subs = {}
        for sub in parsed[-1]:
            key = sub[0]
            value = cls.from_parsed(sub)
            subs[key] = value
        yes = subs[subreferences['yes=']]
        no = subs[subreferences['no=']]
        missing = subs[subreferences['missing=']]
        return cls(variable=variable, threshold=threshold, yes=yes, no=no, missing=missing)
    
    @classmethod
    def from_str(cls, data):
        parsed = cls.parser().parseString(data)
        return cls.from_parsed(parsed)
    
    def __str__(self):
        return 'Node(%s, %f, %s, %s, %s)' % (self.variable, self.threshold, str(self.yes), str(self.no), str(self.missing))
    
    def variables(self):
        variables = set([self.variable])
        if isinstance(self.yes, Node):
            variables |= self.yes.variables()
        if isinstance(self.no, Node):
            variables |= self.no.variables()
        if isinstance(self.missing, Node):
            variables |= self.missing.variables()
        return variables
        
    def expression(self):
        x = RealVariable(self.variable)
        t = RealNumber(self.threshold)
        if isinstance(self.missing, Node):
            missing = self.missing.expression()
        else:
            missing = RealNumber(self.missing)
        if isinstance(self.yes, Node):
            yes = self.yes.expression()
        else:
            yes = RealNumber(self.yes)
        if isinstance(self.no, Node):
            no = self.no.expression()
        else:
            no = RealNumber(self.no)
        return Piecewise((missing, IsNan(x)), (yes, x < t), (no, true))


@sym_predict.register(XGBRegressor)
def sym_predict_xgb_regressor(estimator):
    dump = estimator.booster().get_dump()
    inputs = tuple(map(RealVariable, estimator.booster().feature_names))
    Var = VariableFactory(existing=inputs)
    calls = tuple(map(lambda x: ((Var(),), (x, inputs)), map(lambda x: Function(inputs, tuple(), (x.expression(),)), map(Node.from_str, dump))))
    output = reduce(__add__, map(compose(first, first), calls)) + RealNumber(0.5) # TODO: Why do I have to add 0.5?
    return Function(inputs, calls, (output,))
    
if __name__ == '__main__':
    model = XGBRegressor(n_estimators=2, max_depth=1)
    X, y = make_regression()
    X = DataFrame(X, columns=['x%d' % i for i in range(X.shape[1])])
    model.fit(X, y)
    print(sym_predict(model))
    code = sklearn2code(model, ['predict'], numpy_flat)
    print(code)
    print(model.booster().get_dump()[0])
    module = exec_module('module', code)
    print(module.predict(**X.loc[:10,:]))
    print(model.predict(X.loc[:10,:]))
    1+1

