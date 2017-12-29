from ..dispatching import call_method_or_dispatch, create_registerer

sym_transform_dispatcher = {}
sym_transform = call_method_or_dispatch('sym_transform', sym_transform_dispatcher)
register_sym_transform = create_registerer(sym_transform_dispatcher, 'register_sym_transform')
