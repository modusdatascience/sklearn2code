from sympy.core.numbers import RealNumber
from sympy.functions.elementary.piecewise import Piecewise
from sklearn.tree.tree import DecisionTreeRegressor
from ..base import sym_predict,\
    input_size_from_n_features_
from ..function import Function
from ..base import syms
from sklearn2code.sym.base import input_size

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
    return Piecewise((left_expr, symbols[model.tree_.feature[current_node]] <= model.tree_.threshold[current_node]),
                     (right_expr, symbols[model.tree_.feature[current_node]] > model.tree_.threshold[current_node]),
                     )

@sym_predict.register(DecisionTreeRegressor)
def sym_predict_decision_tree_regressor(estimator):
    n_nodes, n_outputs, n_classes = estimator.tree_.value.shape  # @UnusedVariable
    symbols = syms(estimator)
    result = []
    for output_idx in range(n_outputs):
        result.append(_inner_sym_predict_decision_tree_regressor(estimator, symbols, output_idx=output_idx))
    return Function(symbols, tuple(), tuple(result))

input_size.register(DecisionTreeRegressor, input_size_from_n_features_)
