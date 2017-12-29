from ..dispatching import call_method_or_dispatch, create_registerer

sym_score_to_decision_dispatcher = {}
sym_score_to_decision = call_method_or_dispatch('sym_score_to_decision', sym_score_to_decision_dispatcher)
register_sym_score_to_decision = create_registerer(sym_score_to_decision_dispatcher, 'register_sym_score_to_decision')
