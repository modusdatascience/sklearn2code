from sklearn.tree.tree import DecisionTreeRegressor, DecisionTreeClassifier
from ..base import sym_predict,\
    input_size_from_n_features_
from ..function import Function
from ..base import syms
from ..base import input_size
from ..expression import RealNumber, Piecewise
from sklearn2code.sym.base import sym_predict_proba
from sklearn2code.sym.function import VariableFactory
from itertools import starmap
from sklearn2code.sym.expression import true, Sum, Vec

def _inner_sym_predict_decision_tree_regressor(model, symbols, current_node=0, output_idx=0, class_idx=0):
    left = model.tree_.children_left[current_node]
    right = model.tree_.children_right[current_node]
    if left == -1:
        assert right == -1
        left_expr = RealNumber(model.tree_.value[current_node, output_idx, class_idx])
        right_expr = left_expr
    else:
        left_expr = _inner_sym_predict_decision_tree_regressor(model, symbols, current_node=left, output_idx=output_idx, class_idx=class_idx)
        right_expr = _inner_sym_predict_decision_tree_regressor(model, symbols, current_node=right, output_idx=output_idx, class_idx=class_idx)
    return Piecewise((left_expr, symbols[model.tree_.feature[current_node]] <= RealNumber(model.tree_.threshold[current_node])),
                     (right_expr, symbols[model.tree_.feature[current_node]] > RealNumber(model.tree_.threshold[current_node])),
                     )

# def _inner_sym_predict_decision_tree_classifier(model, symbols, current_node=0, output_idx=0, class_idx=0):


@sym_predict.register(DecisionTreeRegressor)
def sym_predict_decision_tree_regressor(estimator):
    n_nodes, n_outputs, n_classes = estimator.tree_.value.shape  # @UnusedVariable
    symbols = syms(estimator)
    result = []
    for output_idx in range(n_outputs):
        result.append(_inner_sym_predict_decision_tree_regressor(estimator, symbols, output_idx=output_idx))
    return Function(symbols, tuple(), tuple(result))

@sym_predict_proba.register(DecisionTreeClassifier)
def sym_predict_proba_decision_tree_classifier(estimator):
    n_nodes, n_outputs, n_classes = estimator.tree_.value.shape  # @UnusedVariable
    symbols = syms(estimator)
    inputs = syms(estimator)
    inner_result = []
    for output_idx in range(n_outputs):
        inner = list()
        for class_idx in range(n_classes):
            inner.append(_inner_sym_predict_decision_tree_regressor(estimator, symbols, output_idx=output_idx, class_idx=class_idx))
        inner_result.append(inner)
    inner_result_fun = Function(symbols, tuple(), tuple(starmap(Vec, inner_result)))
    Var = VariableFactory(existing=inputs)
    vars_ = tuple(Var() for _ in inner_result)
    summation = Function(vars_, tuple(), tuple(map(Sum, vars_)))
    sums = tuple(Var() for _ in inner_result)
    calls = (
             (vars_, (inner_result_fun, inputs)),
             (sums, (summation, vars_)),
             )
    outputs = tuple(Piecewise((v / s, s > RealNumber(0)), (v, true)) for v, s in zip(vars_, sums))
    return Function(inputs, calls, outputs)
    
#     return Function(symbols, tuple(), tuple(starmap(VectorExpression, result)))
#     if estimator.n_outputs == 1:
#         Var = VariableFactory()
#         vars_ = tuple(Var() for _ in range(len(pre_result.outputs)))
#         total = Function(vars_, 
#                          tuple(), 
#                          (reduce(__add__, vars_),))
#         t = Var()
#         normalize = Function(vars_,
#                              (((t,),(total, vars_)),),
#                              tuple(v / t for v in vars_))

input_size.register(DecisionTreeRegressor, input_size_from_n_features_)
