from operator import add, or_, __or__
from itertools import compress, chain
from six.moves import reduce
from _operator import methodcaller

def assert_parts_are_composable(parts):
    inputs, expressions, target = parts
    try:
        assert set(inputs) >= set(chain(*map(lambda x:x.free_symbols, expressions)))
    except:
        assert set(inputs) >= set(chain(*map(lambda x: set(map(methodcaller('free_symbols'), x)), expressions)))
    if target is not None:
        target_inputs, _, _ = target
        try:
            assert len(target_inputs) == len(expressions) 
            assert_parts_are_composable(target)
        except:
            assert len(target_inputs) == len(expressions) 
            assert_parts_are_composable(target)
            
def double_check(fn):
    def _double_check(*args, **kwargs):
        result = fn(*args, **kwargs)
        try:
            assert_parts_are_composable(result)
        except:
            raise
        return result
    return _double_check

def assemble_parts_into_expressions(parts):
    inputs, expressions, target = parts
    if target is not None:
        target_inputs, target_expressions, target_target = target
        assert len(target_inputs) == len(expressions)
        composed_expressions = [expr.subs(dict(zip(target_inputs, expressions))) for expr in target_expressions]
        return assemble_parts_into_expressions((inputs, composed_expressions, target_target))
    else:
        return inputs, expressions
    
def trim_parts(parts, top=True):
    inputs, expressions, target = parts
    if target is None:
        used_symbols = reduce(__or__, map(lambda x: x.free_symbols, expressions))
        result, index_result = ([inp for inp in inputs if inp in used_symbols], expressions, None), [inp in used_symbols for inp in inputs]
    else:
        target_result, index = trim_parts(target, top=False)
        used_expressions = list(compress(expressions, index))
        used_symbols = reduce(or_, map(lambda x: x.free_symbols, used_expressions), set())
        used_inputs = [inp for inp in inputs if inp in used_symbols]
        new_index = [inp in used_symbols for inp in inputs]
        result, index_result = (used_inputs, used_expressions, target_result), new_index
    if top:
        return result
    else:
        return result, index_result
    
def assemble_parts_into_assignment_pairs_and_outputs(parts):
    _, expressions, target = parts
    result = []
    if target is not None:
        target_inputs, _, _ = target
        assert len(target_inputs) == len(expressions)
        result.extend(zip(target_inputs, expressions))
        target_result, outputs = assemble_parts_into_assignment_pairs_and_outputs(target)
        result.extend(target_result)
        return result, outputs
    else:
        return result, expressions

