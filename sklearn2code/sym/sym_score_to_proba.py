from ..dispatching import call_method_or_dispatch, create_registerer

sym_score_to_proba_dispatcher = {}
sym_score_to_proba = call_method_or_dispatch('sym_score_to_proba', sym_score_to_proba_dispatcher)
register_sym_score_to_proba = create_registerer(sym_score_to_proba_dispatcher, 'register_sym_score_to_proba')
