from toolz.functoolz import curry
from itertools import chain

class covariant(object):
    'Singleton'
cov = covariant
class contravariant(object):
    'Singleton'
con = contravariant
class invariant(object):
    'Singleton'
inv = invariant

class Rep(object):
    def __init__(self, lower, upper, greedy=True):
        self.lower = lower
        self.upper = upper
        self.greedy = greedy
    
    def __contains__(self, count):
        if self.lower is not None and count < self.lower:
            return False
        if self.upper is not None and count > self.upper:
            return False
        return True
    
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (self.__class__ is other.__class__ and self.lower == other.lower and 
                self.upper == other.upper and self.greedy == other.greedy)
    
    def __hash__(self):
        return hash((self.__class__, self.lower, self.upper, self.greedy))
    
def arg_to_rep(arg):
    if arg is None:
        return Rep(1,1)
    elif isinstance(arg, int):
        return Rep(arg, arg)
    elif isinstance(arg, (tuple, list)):
        if len(arg) != 2:
            raise ValueError()
        return Rep(arg[0], arg[1])
    elif isinstance(arg, Rep):
        return arg
    else:
        raise ValueError()

def issub(t1, t2, var):
    if var is invariant:
        return t1 is t2
    elif var is covariant:
        return issubclass(t1, t2)
    elif var is contravariant:
        return issubclass(t2, t1)

class TypeSignature(object):
    def __init__(self, typetups):
        self.typetups = typetups
    
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.__class__ is other.__class__ and self.typetups == other.typetups
    
    def __hash__(self):
        return hash((self.__class__, self.typetups))
    
    @classmethod
    def from_slices(cls, *slices):
        typetups = tuple()
        for slc in slices:
            if isinstance(slc, slice):
                t = slc.start
                rep = arg_to_rep(slc.stop)
                var = slc.step if slc.step is not None else covariant
                typetups += ((t, rep, var),)
            else:
                t = slc
                rep = arg_to_rep(1)
                var = covariant
                typetups += ((t, rep, var),)
        return cls(typetups)
    
    def is_example(self, types):
        it = chain(iter(self.typetups), (None,))
        cur = next(it)
        peek = next(it)
        count = 0
        result = tuple()
        for t in types:
            if cur[1].upper == count:
                if peek is not None:
                    cur = peek
                    peek = next(it)
                    count = 0
                else:
                    return False
            if count not in cur[1]:
                if issub(t, cur[0], cur[2]):
                    result += (cur[0],)
                    count += 1
                    continue
                else:
                    return False
            else:
                if not cur[1].greedy and peek is not None and issub(t, peek[0], peek[2]):
                    cur = peek
                    peek = next(it)
                    count = 0
                    result += (cur[0],)
                    count += 1
                    continue
                elif issub(t, cur[0], cur[2]):
                    result += (cur[0],)
                    count += 1
                    continue
                elif peek is not None and issub(t, peek[0], peek[2]):
                    cur = peek
                    peek = next(it)
                    count = 0
                    result += (cur[0],)
                    continue
                else:
                    return False
        if count not in cur[1]:
            return False
        return result

class DispatcherRegisterer(object):
    def __init__(self, dispatcher):
        self.dispatcher = dispatcher
    
    def __getitem__(self, arg):
        sig = TypeSignature.from_slices(arg)
        def reg(fun):
            self.dispatcher.implementations[sig] = fun
            return fun
        return reg

class AmbiguousDispatchError(Exception):
    pass

class Dispatcher(object):
    def __init__(self, name, doc=''):
        self.__name__ = name
        self.__doc__ = doc
        self.implementations = dict()
    
    @property
    def register(self):
        return DispatcherRegisterer(self)
    
    def dispatch_key(self, *args):
        best = False
        best_sig = None
        for sig in self.implementations.keys():
            current = sig.is_example(tuple(map(type, args)))
            if current:
                if not best or current < best:
                    best = current
                    best_sig = sig
                elif best < current:
                    pass
                else:
                    raise AmbiguousDispatchError()
        return best_sig
    
    def dispatch_value(self, *args):
        key = self.dispatch_key(*args)
        if key:
            return self.implementations[key]
        else:
            raise NotImplementedError()
    
    def __call__(self, *args):
        return self.dispatch_value(*args)(*args)

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
