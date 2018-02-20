from sklearn.ensemble.voting_classifier import VotingClassifier
from ..base import sym_predict, syms
from ..function import cart
from itertools import repeat
from ..function import Function, VariableFactory
from operator import __add__
from six.moves import reduce
from sklearn2code.sym.expression import RealNumber

def sym_weighted_vote(votes, weights):
    return WeightedMode(*reduce(__add__, zip(votes, weights)))

@sym_predict.register(VotingClassifier)
def sym_predict_voting_classifier(estimator):
    inputs = syms(estimator)
    voters = cart(*map(sym_predict, estimator.estimators_))
    if estimator.weights:
        weights = tuple(map(RealNumber, estimator.weights))
    else:
        weights = tuple(repeat(RealNumber(1), len(voters)))
    Var = VariableFactory(existing=inputs)
    votes = tuple(Var() for _ in voters.outputs)
    return Function(inputs=votes, outputs=sym_weighted_vote(votes, weights)).compose(voters)
