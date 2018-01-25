from sklearn2code.sym.base import register_syms, syms, register_input_size,\
    input_size, register_sym_transform, sym_transform, register_sym_predict,\
    sym_predict, register_sym_predict_proba, sym_predict_proba,\
    register_sym_decision_function, sym_decision_function
from sklearn.pipeline import Pipeline
from sklearn2code.sym.function import comp, tupget
from toolz.functoolz import compose



@register_syms(Pipeline)
def syms_pipeline(estimator):
    return syms(estimator.steps[0][1])

@register_input_size(Pipeline)
def input_size_pipeline(estimator):
    return input_size(estimator.steps[0][1])

def sym_transform_intermediate_stages(estimator):
    return comp(*map(compose(sym_transform, tupget(1)), estimator.steps[:-1]))

@register_sym_transform(Pipeline)
def sym_transform_pipeline(estimator):
    return sym_transform(estimator.steps[-1][1]).compose(sym_transform_intermediate_stages(estimator))
    
@register_sym_predict(Pipeline)
def sym_predict_pipeline(estimator):
    return sym_predict(estimator.steps[-1][1]).compose(sym_transform_intermediate_stages(estimator))

@register_sym_predict_proba(Pipeline)
def sym_predict_proba_pipeline(estimator):
    return sym_predict_proba(estimator.steps[-1][1]).compose(sym_transform_intermediate_stages(estimator))

@register_sym_decision_function(Pipeline)
def sym_decision_function_pipeline(estimator):
    return sym_decision_function(estimator.steps[-1][1]).compose(sym_transform_intermediate_stages(estimator))
