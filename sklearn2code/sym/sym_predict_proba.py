from ..dispatching import call_method_or_dispatch, create_registerer

sym_predict_proba_dispatcher = {
                                }
sym_predict_proba = call_method_or_dispatch('sym_predict_proba', sym_predict_proba_dispatcher)
register_sym_predict_proba = create_registerer(sym_predict_proba_dispatcher, 'register_sym_predict_proba')
