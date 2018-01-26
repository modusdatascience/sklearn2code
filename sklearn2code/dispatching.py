from toolz.functoolz import curry

class call_method_or_dispatch(object):
    def __init__(self, method_name, docstring):
        self.__name__ = method_name
        self.dispatcher = dict()
        self.__doc__ = docstring
        
    def __call__(self, estimator, *args, **kwargs):
        try:
            return getattr(estimator, self.__name__)(*args, **kwargs)
        except AttributeError:
            for klass in type(estimator).mro():
                if klass in self.dispatcher:
                    return self.dispatcher[klass](estimator, *args, **kwargs)
            raise NotImplementedError('Class %s does not have an implementation for %s.' % (type(estimator).__name__, self.__name__))
        except:
            raise
    
    @curry
    def register(self, cls, fun):
        self.dispatcher[cls] = fun
        return fun

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
