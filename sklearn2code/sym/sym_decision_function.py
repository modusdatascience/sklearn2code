from ..dispatching import call_method_or_dispatch, create_registerer

sym_decision_function_dispatcher = {}

sym_decision_function = call_method_or_dispatch('sym_decision_function', sym_decision_function_dispatcher)
register_sym_decision_function = create_registerer(sym_decision_function_dispatcher, 'register_sym_decision_function')

