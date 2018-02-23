from sklearn.ensemble.forest import RandomForestRegressor
from ..base import sym_predict, syms
from ..function import VariableFactory, Function
from six.moves import reduce
from itertools import repeat
from operator import __add__, getitem, __truediv__
from toolz.functoolz import curry, flip
from sklearn2code.sym.expression import RealNumber

@sym_predict.register(RandomForestRegressor)
def sym_predict_random_forest_regressor(estimator):
    inputs = syms(estimator)
    Var = VariableFactory(existing=inputs)
    subs = tuple(map(sym_predict, estimator.estimators_))
    calls = tuple((tuple(Var() for _ in range(len(sub.outputs))), (sub, inputs)) for sub in subs)
    outputs = tuple(map(flip(__truediv__)(RealNumber(len(subs))), map(curry(reduce)(__add__), zip(*map(flip(getitem)(0), calls)))))
    return Function(inputs, calls, outputs)

