from sklearn2code.sym.base import syms,\
    input_size, sym_transform,\
    sym_predict, sym_predict_proba,\
    sym_decision_function
from sklearn.pipeline import Pipeline
from sklearn2code.sym.function import comp, tupget
from toolz.functoolz import compose



@syms.register(Pipeline)
def syms_pipeline(estimator):
    return syms(estimator.steps[0][1])

@input_size.register(Pipeline)
def input_size_pipeline(estimator):
    return input_size(estimator.steps[0][1])

def sym_transform_intermediate_stages(estimator):
    return comp(*map(compose(sym_transform, tupget(1)), estimator.steps[:-1]))

@sym_transform.register(Pipeline)
def sym_transform_pipeline(estimator):
    return sym_transform(estimator.steps[-1][1]).compose(sym_transform_intermediate_stages(estimator))
    
@sym_predict.register(Pipeline)
def sym_predict_pipeline(estimator):
    return sym_predict(estimator.steps[-1][1]).compose(sym_transform_intermediate_stages(estimator))

@sym_predict_proba.register(Pipeline)
def sym_predict_proba_pipeline(estimator):
    return sym_predict_proba(estimator.steps[-1][1]).compose(sym_transform_intermediate_stages(estimator))

@sym_decision_function.register(Pipeline)
def sym_decision_function_pipeline(estimator):
    return sym_decision_function(estimator.steps[-1][1]).compose(sym_transform_intermediate_stages(estimator))
