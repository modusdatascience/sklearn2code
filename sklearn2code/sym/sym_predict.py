from ..dispatching import call_method_or_dispatch, create_registerer

sym_predict_dispatcher = {}

sym_predict = call_method_or_dispatch('sym_predict', sym_predict_dispatcher)
register_sym_predict = create_registerer(sym_predict_dispatcher, 'register_sym_predict')

