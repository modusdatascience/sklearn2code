from toolz.functoolz import curry


def call_method_or_dispatch(method_name, dispatcher):
    def _call_method_or_dispatch(estimator, *args, **kwargs):
        try:
            return getattr(estimator, method_name)(*args, **kwargs)
        except AttributeError:
            for klass in type(estimator).mro():
                if klass in dispatcher:
                    return dispatcher[klass](estimator, *args, **kwargs)
            raise
        except:
            raise
    _call_method_or_dispatch.__name__ = method_name
    return _call_method_or_dispatch

def fallback(*args, exception_type=AttributeError):
    def _fallback(*inner_args, **kwargs):
        steps = list(args)
        while steps:
            try:
                return steps.pop(0)(*inner_args, **kwargs)
            except exception_type:
                if not steps:
                    raise
    _fallback.__name__ = args[0].__name__
    return _fallback

def create_registerer(dispatcher, name):
    @curry
    def _register(cls, function):
        dispatcher[cls] = function
        return function
    _register.__name__ = name
    return _register
