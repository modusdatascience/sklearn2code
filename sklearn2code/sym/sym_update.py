from .base import call_method_or_dispatch, fallback, create_registerer
from .sym_transform import sym_transform

sym_update_dispatcher = {}
sym_update = fallback(call_method_or_dispatch('sym_update', sym_update_dispatcher), sym_transform)
register_sym_update = create_registerer(sym_update_dispatcher, 'register_sym_update')