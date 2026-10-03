from functools import reduce, partial
import inspect
import sys
from operator import attrgetter, not_
from importlib import import_module
from types import MethodType

from .utils import no_default

if sys.version_info >= (3, 14):
    import annotationlib  # already imported by inspect


__all__ = ('identity', 'apply', 'thread_first', 'thread_last', 'memoize',
           'compose', 'compose_left', 'pipe', 'complement', 'juxt', 'do',
           'curry', 'flip', 'excepts', '_', 'placeholder')

PYPY = hasattr(sys, 'pypy_version_info')


def identity(x):
    """ Identity function. Return x

    >>> identity(3)
    3
    """
    return x


def apply(*func_and_args, **kwargs):
    """ Applies a function and returns the results

    >>> def double(x): return 2*x
    >>> def inc(x):    return x + 1
    >>> apply(double, 5)
    10

    >>> tuple(map(apply, [double, inc, double], [10, 500, 8000]))
    (20, 501, 16000)
    """
    if not func_and_args:
        raise TypeError('func argument is required')
    func, args = func_and_args[0], func_and_args[1:]
    return func(*args, **kwargs)


def thread_first(val, *forms):
    """ Thread value through a sequence of functions/forms

    >>> def double(x): return 2*x
    >>> def inc(x):    return x + 1
    >>> thread_first(1, inc, double)
    4

    If the function expects more than one input you can specify those inputs
    in a tuple.  The value is used as the first input.

    >>> def add(x, y): return x + y
    >>> def pow(x, y): return x**y
    >>> thread_first(1, (add, 4), (pow, 2))  # pow(add(1, 4), 2)
    25

    So in general
        thread_first(x, f, (g, y, z))
    expands to
        g(f(x), y, z)

    See Also:
        thread_last
    """
    def evalform_front(val, form):
        if callable(form):
            return form(val)
        if isinstance(form, tuple):
            func, args = form[0], form[1:]
            args = (val,) + args
            return func(*args)
    return reduce(evalform_front, forms, val)


def thread_last(val, *forms):
    """ Thread value through a sequence of functions/forms

    >>> def double(x): return 2*x
    >>> def inc(x):    return x + 1
    >>> thread_last(1, inc, double)
    4

    If the function expects more than one input you can specify those inputs
    in a tuple.  The value is used as the last input.

    >>> def add(x, y): return x + y
    >>> def pow(x, y): return x**y
    >>> thread_last(1, (add, 4), (pow, 2))  # pow(2, add(4, 1))
    32

    So in general
        thread_last(x, f, (g, y, z))
    expands to
        g(y, z, f(x))

    >>> def iseven(x):
    ...     return x % 2 == 0
    >>> list(thread_last([1, 2, 3], (map, inc), (filter, iseven)))
    [2, 4]

    See Also:
        thread_first
    """
    def evalform_back(val, form):
        if callable(form):
            return form(val)
        if isinstance(form, tuple):
            func, args = form[0], form[1:]
            args = args + (val,)
            return func(*args)
    return reduce(evalform_back, forms, val)


def instanceproperty(fget=None, fset=None, fdel=None, doc=None, classval=None):
    """ Like @property, but returns ``classval`` when used as a class attribute

    >>> class MyClass(object):
    ...     '''The class docstring'''
    ...     @instanceproperty(classval=__doc__)
    ...     def __doc__(self):
    ...         return 'An object docstring'
    ...     @instanceproperty
    ...     def val(self):
    ...         return 42
    ...
    >>> MyClass.__doc__
    'The class docstring'
    >>> MyClass.val is None
    True
    >>> obj = MyClass()
    >>> obj.__doc__
    'An object docstring'
    >>> obj.val
    42
    """
    if fget is None:
        return partial(instanceproperty, fset=fset, fdel=fdel, doc=doc,
                       classval=classval)
    return InstanceProperty(fget=fget, fset=fset, fdel=fdel, doc=doc,
                            classval=classval)


class InstanceProperty(property):
    """ Like @property, but returns ``classval`` when used as a class attribute

    Should not be used directly.  Use ``instanceproperty`` instead.
    """
    def __init__(self, fget=None, fset=None, fdel=None, doc=None,
                 classval=None):
        self.classval = classval
        property.__init__(self, fget=fget, fset=fset, fdel=fdel, doc=doc)

    def __get__(self, obj, type=None):
        if obj is None:
            return self.classval
        return property.__get__(self, obj, type)

    def __reduce__(self):
        state = (self.fget, self.fset, self.fdel, self.__doc__, self.classval)
        return InstanceProperty, state


class _PlaceholderType:
    """ Explicit placeholder for a deferred positional or keyword argument.

    Use the singleton ``_`` (or its alias ``placeholder``) to reserve a
    position when currying.  Later calls fill the reserved slots in order.

    >>> def f(a, b, c, d):
    ...     return (a, b, c, d)
    >>> g = curry(f)(_, 2, _, 4)
    >>> g(1, 3)
    (1, 2, 3, 4)

    Positional arguments fill the empty slots from left to right; keyword
    arguments bind through the function's signature and may fill a slot by
    name.  Binding the same parameter twice is an error.

    While slots remain unfilled, calling the curry returns a new curry; use
    ``call`` to force evaluation (it raises ``TypeError`` on unfilled slots).

    >>> g = curry(f)(_, 2, _, 4)
    >>> g.call(1, 3)
    (1, 2, 3, 4)
    """
    __slots__ = ()

    def __repr__(self):
        return '_'

    def __reduce__(self):
        return (_get_placeholder, ())


def _get_placeholder():
    return _


#: Singleton placeholder used to leave an argument slot open when currying.
_ = _PlaceholderType()
#: Descriptive alias for :data:`_`.
placeholder = _


class curry:
    """ Curry a callable function

    Enables partial application of arguments through calling a function with an
    incomplete set of arguments.

    >>> def mul(x, y):
    ...     return x * y
    >>> mul = curry(mul)

    >>> double = mul(2)
    >>> double(10)
    20

    Also supports keyword arguments

    >>> @curry                  # Can use curry as a decorator
    ... def f(x, y, a=10):
    ...     return a * (x + y)

    >>> add = f(a=1)
    >>> add(2, 3)
    5

    The placeholder ``_`` leaves an argument slot open.  Later positional
    arguments fill the open slots in order; any extra positional arguments
    are appended after them.

    >>> def report(a, b, c, d):
    ...     return (a, b, c, d)
    >>> g = curry(report)(_, 2, _, 4)
    >>> g(1, 3)
    (1, 2, 3, 4)
    >>> g(1, 3, 5)
    (1, 2, 3, 4, 5)

    Keyword arguments bind through the Python signature and can close a slot
    by name, but the same parameter can never be bound twice.

    >>> h = curry(report)(_, 2, _, 4)
    >>> h(1, c=3)
    (1, 2, 3, 4)
    >>> h(3, a=1)
    Traceback (most recent call last):
    ...
    TypeError: ...

    As long as slots remain open, calling returns a new curry object.

    >>> type(g(1)) is curry
    True

    See Also:
        toolz.curried - namespace of curried functions
                        https://toolz.readthedocs.io/en/latest/curry.html
    """
    def __init__(self, *args, **kwargs):
        if not args:
            raise TypeError('__init__() takes at least 2 arguments (1 given)')
        func, args = args[0], args[1:]
        if not callable(func):
            raise TypeError("Input must be callable")

        # curry- or functools.partial-like object?  Unpack and merge arguments
        if (
            hasattr(func, 'func')
            and hasattr(func, 'args')
            and hasattr(func, 'keywords')
            and isinstance(func.args, tuple)
        ):
            merged_kwargs = {}
            if func.keywords:
                merged_kwargs.update(func.keywords)
            merged_kwargs.update(kwargs)
            kwargs = merged_kwargs
            args = func.args + args
            func = func.func

        self._curry_func = func
        self._curry_args = args
        self._curry_keywords = kwargs
        self._sigspec = None
        self._has_unknown_args = None
        self._has_holes = any(arg is _ for arg in args) or any(
            value is _ for value in kwargs.values())
        if self._has_holes:
            self._validate_holes()

        self.__doc__ = getattr(func, '__doc__', None)
        self.__name__ = getattr(func, '__name__', '<curry>')
        self.__module__ = getattr(func, '__module__', None)
        self.__qualname__ = getattr(func, '__qualname__', None)

    @instanceproperty
    def func(self):
        return self._curry_func

    @instanceproperty
    def __signature__(self):
        sig = inspect.signature(self.func)
        args = self.args or ()
        keywords = self.keywords or {}
        if is_partial_args(self.func, args, keywords, sig) is False:
            raise TypeError('curry object has incorrect arguments')

        params = list(sig.parameters.values())

        # Walk the stored positional pattern in lockstep with the
        # signature's positional parameters.  Concrete values that precede
        # any open slot are consumed; an open slot keeps its parameter, so
        # the remaining signature still asks for it.  Parameters fixed by
        # later concrete values are exposed as keyword-only.
        param_index = 0
        keep_from = None
        forced_kwonly = {}
        for i, value in enumerate(args):
            while (param_index < len(params)
                   and params[param_index].kind not in
                   (inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL)):
                param_index += 1
            if param_index >= len(params):
                break
            param = params[param_index]
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                keep_from = param_index
                break
            if value is _:
                if keep_from is None:
                    keep_from = param_index
            elif param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                forced_kwonly[param.name] = value
            param_index += 1
        if keep_from is None:
            keep_from = min(param_index, len(params))

        kwonly = False
        newparams = []
        for param in params[keep_from:]:
            kind = param.kind
            default = param.default
            if kind == inspect.Parameter.VAR_KEYWORD:
                pass
            elif kind == inspect.Parameter.VAR_POSITIONAL:
                if kwonly:
                    continue
            elif param.name in keywords and keywords[param.name] is not _:
                default = keywords[param.name]
                kind = inspect.Parameter.KEYWORD_ONLY
                kwonly = True
            elif param.name in forced_kwonly:
                default = forced_kwonly[param.name]
                kind = inspect.Parameter.KEYWORD_ONLY
                kwonly = True
            else:
                if kwonly:
                    kind = inspect.Parameter.KEYWORD_ONLY
                if default is inspect.Parameter.empty:
                    default = no_default
            newparams.append(param.replace(default=default, kind=kind))

        return sig.replace(parameters=newparams)

    @instanceproperty
    def args(self):
        return self._curry_args

    @instanceproperty
    def keywords(self):
        return self._curry_keywords

    @instanceproperty
    def func_name(self):
        return self.__name__

    def __str__(self):
        return str(self.func)

    def __repr__(self):
        return repr(self.func)

    def __hash__(self):
        return hash((self.func, self.args,
                     frozenset(self.keywords.items()) if self.keywords
                     else None))

    def __eq__(self, other):
        return (isinstance(other, curry) and self.func == other.func and
                self.args == other.args and self.keywords == other.keywords)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __call__(self, *args, **kwargs):
        if self._has_holes or any(arg is _ for arg in args) or any(
                value is _ for value in kwargs.values()):
            args, kwargs, state = self._merge_holes(args, kwargs)
            if state != 'complete':
                return self._bind_merged(args, kwargs, state != 'unfilled')
            if kwargs:
                return self._curry_func(*args, **kwargs)
            return self._curry_func(*args)
        try:
            if self.keywords:
                return self._curry_func(*self.args, *args,
                                  **dict(self.keywords, **kwargs))
            return self._curry_func(*self.args, *args, **kwargs)
        except TypeError as exc:
            if self._should_curry(args, kwargs, exc):
                return self.bind(*args, **kwargs)
            raise

    def _should_curry(self, args, kwargs, exc=None):
        func = self.func
        if self.keywords:
            kwargs = dict(self.keywords, **kwargs)
        args = self.args + args
        if self._sigspec is None:
            sigspec = self._sigspec = _sigs.signature_or_spec(func)
            self._has_unknown_args = has_varargs(func, sigspec) is not False
        else:
            sigspec = self._sigspec

        if is_partial_args(func, args, kwargs, sigspec) is False:
            # Nothing can make the call valid
            return False
        elif self._has_unknown_args:
            # The call may be valid and raised a TypeError, but we curry
            # anyway because the function may have `*args`.  This is useful
            # for decorators with signature `func(*args, **kwargs)`.
            return True
        elif not is_valid_args(func, args, kwargs, sigspec):
            # Adding more arguments may make the call valid
            return True
        else:
            # There was a genuine TypeError
            return False

    def bind(self, *args, **kwargs):
        if self._has_holes or any(arg is _ for arg in args) or any(
                value is _ for value in kwargs.values()):
            args, kwargs, state = self._merge_holes(args, kwargs)
            return self._bind_merged(args, kwargs, state != 'complete')
        return type(self)(self, *args, **kwargs)

    def call(self, *args, **kwargs):
        if self._has_holes or any(arg is _ for arg in args) or any(
                value is _ for value in kwargs.values()):
            args, kwargs, state = self._merge_holes(args, kwargs)
            if state == 'unfilled':
                raise TypeError(
                    'Not enough arguments to fill the placeholder slots')
        elif self.keywords:
            kwargs = dict(self.keywords, **kwargs)
            args = self.args + args
        else:
            args = self.args + args
        if kwargs:
            return self._curry_func(*args, **kwargs)
        return self._curry_func(*args)

    def _bind_merged(self, args, kwargs, has_holes):
        obj = type(self).__new__(type(self))
        obj._curry_func = self._curry_func
        obj._curry_args = args
        obj._curry_keywords = kwargs
        obj._has_holes = bool(has_holes)
        obj.__doc__ = self.__doc__
        obj.__name__ = self.__name__
        obj.__module__ = self.__module__
        obj.__qualname__ = self.__qualname__
        obj._sigspec = self._sigspec
        obj._has_unknown_args = self._has_unknown_args
        return obj

    def _merge_holes(self, args, kwargs):
        """Merge new args/kwargs into a pattern that may contain ``_``.

        New positional arguments fill open slots from left to right; values
        left over are appended after the existing positional pattern.  New
        keyword arguments first fill positional-or-keyword slots that are
        still open (matched through the function signature) and are then
        merged into the keyword mapping.  Raises ``TypeError`` if the merge
        binds the same parameter more than once.
        """
        pattern = list(self.args)
        keywords = dict(self.keywords)

        new_args = list(args)
        n_open_before = sum(value is _ for value in pattern)
        for i, value in enumerate(pattern):
            if value is _ and new_args:
                pattern[i] = new_args.pop(0)
        n_leftovers = len(new_args)
        pattern.extend(new_args)

        claimed = self._claim_holes_by_keyword(pattern, keywords, kwargs)

        for name, value in kwargs.items():
            if name in keywords and keywords[name] is not _:
                if value is not _:
                    raise TypeError(
                        f"got multiple values for keyword argument {name!r}")

            keywords[name] = value

        args = tuple(pattern)
        has_holes = any(value is _ for value in args) or any(
            value is _ for value in keywords.values())
        self._validate_hole_pattern(args, keywords, claimed)
        if has_holes:
            return args, keywords, 'unfilled'
        if (n_open_before > 0 and n_leftovers > 0
                and self._has_varargs(args, kwargs)):
            # Every placeholder was filled and extra positional arguments
            # spilled into ``*args``; stay curried, mirroring the usual
            # behaviour for signatures with variadic positional arguments.
            return args, keywords, 'curried'
        return args, keywords, 'complete'

    def _has_varargs(self, args, kwargs):
        sigspec = self._get_sigspec()
        if isinstance(sigspec, inspect.Signature):
            return any(p.kind == inspect.Parameter.VAR_POSITIONAL
                       for p in sigspec.parameters.values())
        return self._has_unknown_args or False

    def _claim_holes_by_keyword(self, pattern, keywords, new_kwargs):
        """Let keyword arguments fill open positional-or-keyword slots."""
        sigspec = self._get_sigspec()
        claimed = set()
        if not isinstance(sigspec, inspect.Signature):
            return claimed
        pos_names = [
            p.name for p in sigspec.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        for name, value in list(new_kwargs.items()):
            if value is _ or name not in pos_names:
                continue
            target = pos_names.index(name)
            if target >= len(pattern) or pattern[target] is not _:
                continue
            if name in keywords and keywords[name] is not _:
                raise TypeError(
                    f"got multiple values for keyword argument {name!r}")
            pattern[target] = value
            claimed.add(target)
            del new_kwargs[name]
        return claimed

    def _get_sigspec(self):
        if self._sigspec is None:
            self._sigspec = _sigs.signature_or_spec(self._curry_func)
            self._has_unknown_args = has_varargs(
                self._curry_func, self._sigspec) is not False
        return self._sigspec

    def _validate_holes(self):
        self._validate_hole_pattern(self._curry_args, self._curry_keywords, set())

    def _validate_hole_pattern(self, args, kwargs, claimed):
        """Ensure the pattern could still be completed by later arguments."""
        sigspec = self._get_sigspec()
        if not isinstance(sigspec, inspect.Signature):
            return
        dummy = object()
        check_args = tuple(
            dummy if (value is _ and i not in claimed) else value
            for i, value in enumerate(args))
        check_kwargs = {
            name: (dummy if value is _ else value)
            for name, value in kwargs.items()}
        sigspec.bind_partial(*check_args, **check_kwargs)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return curry(self, instance)

    def __reduce__(self):
        func = self.func
        modname = getattr(func, '__module__', None)
        qualname = getattr(func, '__qualname__', None)
        if qualname is None:  # pragma: no cover
            qualname = getattr(func, '__name__', None)
        is_decorated = None
        if modname and qualname:
            attrs = []
            obj = import_module(modname)
            for attr in qualname.split('.'):
                if isinstance(obj, curry):
                    attrs.append('func')
                    obj = obj.func
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
                attrs.append(attr)
            if isinstance(obj, curry) and obj.func is func:
                is_decorated = obj is self
                qualname = '.'.join(attrs)
                func = f'{modname}:{qualname}'

        # functools.partial objects can't be pickled
        userdict = tuple((k, v) for k, v in self.__dict__.items()
                         if k not in ('_curry_func', '_curry_args', '_curry_keywords',
                                      '_sigspec'))
        state = (type(self), func, self.args, self.keywords, userdict,
                 is_decorated)
        return _restore_curry, state


def _restore_curry(cls, func, args, kwargs, userdict, is_decorated):
    if isinstance(func, str):
        modname, qualname = func.rsplit(':', 1)
        obj = import_module(modname)
        for attr in qualname.split('.'):
            obj = getattr(obj, attr)
        if is_decorated:
            return obj
        func = obj.func
    obj = cls(func, *args, **(kwargs or {}))
    obj.__dict__.update(userdict)
    return obj


@curry
def memoize(func, cache=None, key=None):
    """ Cache a function's result for speedy future evaluation

    Considerations:
        Trades memory for speed.
        Only use on pure functions.

    >>> def add(x, y):  return x + y
    >>> add = memoize(add)

    Or use as a decorator

    >>> @memoize
    ... def add(x, y):
    ...     return x + y

    Use the ``cache`` keyword to provide a dict-like object as an initial cache

    >>> @memoize(cache={(1, 2): 3})
    ... def add(x, y):
    ...     return x + y

    Note that the above works as a decorator because ``memoize`` is curried.

    It is also possible to provide a ``key(args, kwargs)`` function that
    calculates keys used for the cache, which receives an ``args`` tuple and
    ``kwargs`` dict as input, and must return a hashable value.  However,
    the default key function should be sufficient most of the time.

    >>> # Use key function that ignores extraneous keyword arguments
    >>> @memoize(key=lambda args, kwargs: args)
    ... def add(x, y, verbose=False):
    ...     if verbose:
    ...         print('Calculating %s + %s' % (x, y))
    ...     return x + y
    """
    if cache is None:
        cache = {}

    try:
        may_have_kwargs = has_keywords(func) is not False
        # Is unary function (single arg, no variadic argument or keywords)?
        is_unary = is_arity(1, func)
    except TypeError:  # pragma: no cover
        may_have_kwargs = True
        is_unary = False

    if key is None:
        if is_unary:
            def key(args, kwargs):
                return args[0]
        elif may_have_kwargs:
            def key(args, kwargs):
                return (
                    args or None,
                    frozenset(kwargs.items()) if kwargs else None,
                )
        else:
            def key(args, kwargs):
                return args

    def memof(*args, **kwargs):
        k = key(args, kwargs)
        try:
            return cache[k]
        except TypeError:
            raise TypeError("Arguments to memoized function must be hashable")
        except KeyError:
            cache[k] = result = func(*args, **kwargs)
            return result

    try:
        memof.__name__ = func.__name__
    except AttributeError:
        pass
    memof.__doc__ = func.__doc__
    memof.__wrapped__ = func
    return memof


class _InstanceAnnotations(dict):
    """ Descriptor for an ``__annotations__`` computed by ``fget`` on instances

    Accessed on the class, this is an empty dict, because ``typing`` and
    ``inspect`` require a class's ``__annotations__`` to be a dict (before
    Python 3.14 they read it directly from the class ``__dict__``).
    """
    __slots__ = ('fget',)

    def __init__(self, fget):
        self.fget = fget

    def __get__(self, obj, type=None):
        return self if obj is None else self.fget(obj)


class Compose:
    """ A composition of functions

    See Also:
        compose
    """
    __slots__ = 'first', 'funcs'

    def __init__(self, funcs):
        funcs = tuple(reversed(funcs))
        self.first = funcs[0]
        self.funcs = funcs[1:]

    def __call__(self, *args, **kwargs):
        ret = self.first(*args, **kwargs)
        for f in self.funcs:
            ret = f(ret)
        return ret

    def __getstate__(self):
        return self.first, self.funcs

    def __setstate__(self, state):
        self.first, self.funcs = state

    @instanceproperty(classval=__doc__)
    def __doc__(self):
        def composed_doc(*fs):
            """Generate a docstring for the composition of fs.
            """
            if not fs:
                # Argument name for the docstring.
                return '*args, **kwargs'

            return f'{fs[0].__name__}({composed_doc(*fs[1:])})'

        try:
            return (
                'lambda *args, **kwargs: ' +
                composed_doc(*reversed((self.first,) + self.funcs))
            )
        except AttributeError:
            # One of our callables does not have a `__name__`, whatever.
            return 'A composition of functions'

    @property
    def __name__(self):
        try:
            return '_of_'.join(
                f.__name__ for f in reversed((self.first,) + self.funcs)
            )
        except AttributeError:
            return type(self).__name__

    def __repr__(self):
        return '{.__class__.__name__}{!r}'.format(
            self, tuple(reversed((self.first, ) + self.funcs)))

    def __eq__(self, other):
        if isinstance(other, Compose):
            return other.first == self.first and other.funcs == self.funcs
        return NotImplemented

    def __ne__(self, other):
        equality = self.__eq__(other)
        return NotImplemented if equality is NotImplemented else not equality

    def __hash__(self):
        return hash(self.first) ^ hash(self.funcs)

    # Mimic the descriptor behavior of python functions.
    # i.e. let Compose be called as a method when bound to a class.
    # adapted from
    # docs.python.org/3/howto/descriptor.html#functions-and-methods
    def __get__(self, obj, objtype=None):
        return self if obj is None else MethodType(self, obj)

    # introspection with Signature is only possible from py3.3+
    @instanceproperty
    def __signature__(self):
        base = inspect.signature(self.first)
        last = inspect.signature(self.funcs[-1])
        return base.replace(return_annotation=last.return_annotation)

    __wrapped__ = instanceproperty(attrgetter('first'))

    def _combined_annotations(self, annotation_format=None):
        """ Combined type annotations for the composed callable.

        Parameter annotations come from the first function applied in the
        composition, and the return annotation comes from the last function
        applied, as ``inspect.signature`` reports them (so this agrees with
        ``__signature__``).  Functions without a signature contribute nothing.

        ``annotation_format`` (Python 3.14+) is passed to
        ``inspect.signature``.  With ``FORWARDREF`` or ``STRING``, which must
        not raise ``NameError``, a function whose ``__signature__`` evaluates
        annotations eagerly (e.g. ``curry``) and fails also contributes
        nothing.

        String annotations are returned as-is; ``typing.get_type_hints``
        evaluates them in the namespace of the first function.
        """
        if annotation_format is None:
            kwargs = {}
            lazy = False
        else:
            kwargs = {'annotation_format': annotation_format}
            lazy = annotation_format != annotationlib.Format.VALUE

        def annotations_of(func):
            if isinstance(func, Compose):
                return func._combined_annotations(annotation_format)
            try:
                sig = inspect.signature(func, **kwargs)
            except (TypeError, ValueError):
                return {}
            except NameError:
                if not lazy:
                    raise
                return {}
            rv = {name: param.annotation
                  for name, param in sig.parameters.items()
                  if param.annotation is not param.empty}
            if sig.return_annotation is not sig.empty:
                rv['return'] = sig.return_annotation
            return rv

        annotations = annotations_of(self.first)
        annotations.pop('return', None)
        last = self.funcs[-1] if self.funcs else self.first
        last_annotations = annotations_of(last)
        if 'return' in last_annotations:
            annotations['return'] = last_annotations['return']
        return annotations

    # Lets ``typing.get_type_hints`` and ``inspect.get_annotations`` see the
    # same types as ``__signature__``.
    __annotations__ = _InstanceAnnotations(_combined_annotations)

    if sys.version_info >= (3, 14):
        def _get_annotate(self):
            def annotate(format):
                Format = annotationlib.Format
                if format == Format.STRING:
                    # Some ``__signature__`` (e.g. ``curry``) ignore the
                    # format and give values, so stringify what we got.
                    return annotationlib.annotations_to_string(
                        self._combined_annotations(format)
                    )
                if format in (Format.VALUE, Format.FORWARDREF):
                    return self._combined_annotations(format)
                raise NotImplementedError(format)
            return annotate

        # PEP 649/749: support the other annotation formats (``FORWARDREF``,
        # ``STRING``) and let ``functools.wraps`` copy the annotations.
        # Not ``def __annotate__``: coverage.py never traces functions with
        # that name, since it assumes they are compiler-generated.
        __annotate__ = instanceproperty(_get_annotate)


def compose(*funcs):
    """ Compose functions to operate in series.

    Returns a function that applies other functions in sequence.

    Functions are applied from right to left so that
    ``compose(f, g, h)(x, y)`` is the same as ``f(g(h(x, y)))``.

    If no arguments are provided, the identity function (f(x) = x) is returned.

    >>> inc = lambda i: i + 1
    >>> compose(str, inc)(3)
    '4'

    See Also:
        compose_left
        pipe
    """
    if not funcs:
        return identity
    if len(funcs) == 1:
        return funcs[0]
    else:
        return Compose(funcs)


def compose_left(*funcs):
    """ Compose functions to operate in series.

    Returns a function that applies other functions in sequence.

    Functions are applied from left to right so that
    ``compose_left(f, g, h)(x, y)`` is the same as ``h(g(f(x, y)))``.

    If no arguments are provided, the identity function (f(x) = x) is returned.

    >>> inc = lambda i: i + 1
    >>> compose_left(inc, str)(3)
    '4'

    See Also:
        compose
        pipe
    """
    return compose(*reversed(funcs))


def pipe(data, *funcs):
    """ Pipe a value through a sequence of functions

    I.e. ``pipe(data, f, g, h)`` is equivalent to ``h(g(f(data)))``

    We think of the value as progressing through a pipe of several
    transformations, much like pipes in UNIX

    ``$ cat data | f | g | h``

    >>> double = lambda i: 2 * i
    >>> pipe(3, double, str)
    '6'

    See Also:
        compose
        compose_left
        thread_first
        thread_last
    """
    for func in funcs:
        data = func(data)
    return data


def complement(func):
    """ Convert a predicate function to its logical complement.

    In other words, return a function that, for inputs that normally
    yield True, yields False, and vice-versa.

    >>> def iseven(n): return n % 2 == 0
    >>> isodd = complement(iseven)
    >>> iseven(2)
    True
    >>> isodd(2)
    False
    """
    return compose(not_, func)


class juxt:
    """ Creates a function that calls several functions with the same arguments

    Takes several functions and returns a function that applies its arguments
    to each of those functions then returns a tuple of the results.

    Name comes from juxtaposition: the fact of two things being seen or placed
    close together with contrasting effect.

    >>> inc = lambda x: x + 1
    >>> double = lambda x: x * 2
    >>> juxt(inc, double)(10)
    (11, 20)
    >>> juxt([inc, double])(10)
    (11, 20)
    """
    __slots__ = ['funcs']

    def __init__(self, *funcs):
        if len(funcs) == 1 and not callable(funcs[0]):
            funcs = funcs[0]
        self.funcs = tuple(funcs)

    def __call__(self, *args, **kwargs):
        return tuple(func(*args, **kwargs) for func in self.funcs)

    def __getstate__(self):
        return self.funcs

    def __setstate__(self, state):
        self.funcs = state


def do(func, x):
    """ Runs ``func`` on ``x``, returns ``x``

    Because the results of ``func`` are not returned, only the side
    effects of ``func`` are relevant.

    Logging functions can be made by composing ``do`` with a storage function
    like ``list.append`` or ``file.write``

    >>> from toolz import compose
    >>> from toolz.curried import do

    >>> log = []
    >>> inc = lambda x: x + 1
    >>> inc = compose(inc, do(log.append))
    >>> inc(1)
    2
    >>> inc(11)
    12
    >>> log
    [1, 11]
    """
    func(x)
    return x


@curry
def flip(func, a, b):
    """ Call the function call with the arguments flipped

    This function is curried.

    >>> def div(a, b):
    ...     return a // b
    ...
    >>> flip(div, 2, 6)
    3
    >>> div_by_two = flip(div, 2)
    >>> div_by_two(4)
    2

    This is particularly useful for built in functions and functions defined
    in C extensions that accept positional only arguments. For example:
    isinstance, issubclass.

    >>> data = [1, 'a', 'b', 2, 1.5, object(), 3]
    >>> only_ints = list(filter(flip(isinstance, int), data))
    >>> only_ints
    [1, 2, 3]
    """
    return func(b, a)


def return_none(exc):
    """ Returns None.
    """
    return None


class excepts:
    """A wrapper around a function to catch exceptions and
    dispatch to a handler.

    This is like a functional try/except block, in the same way that
    ifexprs are functional if/else blocks.

    Examples
    --------
    >>> excepting = excepts(
    ...     ValueError,
    ...     lambda a: [1, 2].index(a),
    ...     lambda _: -1,
    ... )
    >>> excepting(1)
    0
    >>> excepting(3)
    -1

    Multiple exceptions and default except clause.

    >>> excepting = excepts((IndexError, KeyError), lambda a: a[0])
    >>> excepting([])
    >>> excepting([1])
    1
    >>> excepting({})
    >>> excepting({0: 1})
    1
    """
    def __init__(self, exc, func, handler=return_none):
        self.exc = exc
        self.func = func
        self.handler = handler

    def __call__(self, *args, **kwargs):
        try:
            return self.func(*args, **kwargs)
        except self.exc as e:
            return self.handler(e)

    @instanceproperty(classval=__doc__)
    def __doc__(self):
        from textwrap import dedent

        exc = self.exc
        try:
            if isinstance(exc, tuple):
                exc_name = '(%s)' % ', '.join(
                    map(attrgetter('__name__'), exc),
                )
            else:
                exc_name = exc.__name__

            return dedent(
                """\
                A wrapper around {inst.func.__name__!r} that will except:
                {exc}
                and handle any exceptions with {inst.handler.__name__!r}.

                Docs for {inst.func.__name__!r}:
                {inst.func.__doc__}

                Docs for {inst.handler.__name__!r}:
                {inst.handler.__doc__}
                """
            ).format(
                inst=self,
                exc=exc_name,
            )
        except AttributeError:
            return type(self).__doc__

    @property
    def __name__(self):
        exc = self.exc
        try:
            if isinstance(exc, tuple):
                exc_name = '_or_'.join(map(attrgetter('__name__'), exc))
            else:
                exc_name = exc.__name__
            return f'{self.func.__name__}_excepting_{exc_name}'
        except AttributeError:
            return 'excepting'


def _check_sigspec(sigspec, func, builtin_func, *builtin_args):
    if sigspec is None:
        try:
            sigspec = inspect.signature(func)
        except (ValueError, TypeError) as e:
            sigspec = e
    if isinstance(sigspec, ValueError):
        return None, builtin_func(*builtin_args)
    elif not isinstance(sigspec, inspect.Signature):
        if (
            func in _sigs.signatures
            and (
                hasattr(func, '__signature__')
                and hasattr(func.__signature__, '__get__')
            )
        ):
            val = builtin_func(*builtin_args)
            return None, val
        return None, False
    return sigspec, None


if PYPY:  # pragma: no cover
    _check_sigspec_orig = _check_sigspec

    def _check_sigspec(sigspec, func, builtin_func, *builtin_args):
        # PyPy may lie, so use our registry for builtins instead
        if func in _sigs.signatures:
            val = builtin_func(*builtin_args)
            return None, val
        return _check_sigspec_orig(sigspec, func, builtin_func, *builtin_args)


_check_sigspec.__doc__ = """ \
Private function to aid in introspection compatibly across Python versions.

If a callable doesn't have a signature (Python 3) or an argspec (Python 2),
the signature registry in toolz._signatures is used.
"""


def num_required_args(func, sigspec=None):
    sigspec, rv = _check_sigspec(sigspec, func, _sigs._num_required_args,
                                 func)
    if sigspec is None:
        return rv
    return sum(1 for p in sigspec.parameters.values()
               if p.default is p.empty
               and p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY))


def has_varargs(func, sigspec=None):
    sigspec, rv = _check_sigspec(sigspec, func, _sigs._has_varargs, func)
    if sigspec is None:
        return rv
    return any(p.kind == p.VAR_POSITIONAL
               for p in sigspec.parameters.values())


def has_keywords(func, sigspec=None):
    sigspec, rv = _check_sigspec(sigspec, func, _sigs._has_keywords, func)
    if sigspec is None:
        return rv
    return any(p.default is not p.empty
               or p.kind in (p.KEYWORD_ONLY, p.VAR_KEYWORD)
               for p in sigspec.parameters.values())


def is_valid_args(func, args, kwargs, sigspec=None):
    sigspec, rv = _check_sigspec(sigspec, func, _sigs._is_valid_args,
                                 func, args, kwargs)
    if sigspec is None:
        return rv
    try:
        sigspec.bind(*args, **kwargs)
    except TypeError:
        return False
    return True


def is_partial_args(func, args, kwargs, sigspec=None):
    sigspec, rv = _check_sigspec(sigspec, func, _sigs._is_partial_args,
                                 func, args, kwargs)
    if sigspec is None:
        return rv
    try:
        sigspec.bind_partial(*args, **kwargs)
    except TypeError:
        return False
    return True


def is_arity(n, func, sigspec=None):
    """ Does a function have only n positional arguments?

    This function relies on introspection and does not call the function.
    Returns None if validity can't be determined.

    >>> def f(x):
    ...     return x
    >>> is_arity(1, f)
    True
    >>> def g(x, y=1):
    ...     return x + y
    >>> is_arity(1, g)
    False
    """
    sigspec, rv = _check_sigspec(sigspec, func, _sigs._is_arity, n, func)
    if sigspec is None:
        return rv
    num = num_required_args(func, sigspec)
    if num is not None:
        num = num == n
        if not num:
            return False
    varargs = has_varargs(func, sigspec)
    if varargs:
        return False
    keywords = has_keywords(func, sigspec)
    if keywords:
        return False
    if num is None or varargs is None or keywords is None:  # pragma: no cover
        return None
    return True


num_required_args.__doc__ = """ \
Number of required positional arguments

    This function relies on introspection and does not call the function.
    Returns None if validity can't be determined.

    >>> def f(x, y, z=3):
    ...     return x + y + z
    >>> num_required_args(f)
    2
    >>> def g(*args, **kwargs):
    ...     pass
    >>> num_required_args(g)
    0
    """

has_varargs.__doc__ = """ \
Does a function have variadic positional arguments?

    This function relies on introspection and does not call the function.
    Returns None if validity can't be determined.

    >>> def f(*args):
    ...    return args
    >>> has_varargs(f)
    True
    >>> def g(**kwargs):
    ...    return kwargs
    >>> has_varargs(g)
    False
    """

has_keywords.__doc__ = """ \
Does a function have keyword arguments?

    This function relies on introspection and does not call the function.
    Returns None if validity can't be determined.

    >>> def f(x, y=0):
    ...     return x + y

    >>> has_keywords(f)
    True
    """

is_valid_args.__doc__ = """ \
Is ``func(*args, **kwargs)`` a valid function call?

    This function relies on introspection and does not call the function.
    Returns None if validity can't be determined.

    >>> def add(x, y):
    ...     return x + y

    >>> is_valid_args(add, (1,), {})
    False
    >>> is_valid_args(add, (1, 2), {})
    True
    >>> is_valid_args(map, (), {})
    False

    **Implementation notes**
    Python 2 relies on ``inspect.getargspec``, which only works for
    user-defined functions.  Python 3 uses ``inspect.signature``, which
    works for many more types of callables.

    Many builtins in the standard library are also supported.
    """

is_partial_args.__doc__ = """ \
Can partial(func, *args, **kwargs)(*args2, **kwargs2) be a valid call?

    Returns True *only* if the call is valid or if it is possible for the
    call to become valid by adding more positional or keyword arguments.

    This function relies on introspection and does not call the function.
    Returns None if validity can't be determined.

    >>> def add(x, y):
    ...     return x + y

    >>> is_partial_args(add, (1,), {})
    True
    >>> is_partial_args(add, (1, 2), {})
    True
    >>> is_partial_args(add, (1, 2, 3), {})
    False
    >>> is_partial_args(map, (), {})
    True

    **Implementation notes**
    Python 2 relies on ``inspect.getargspec``, which only works for
    user-defined functions.  Python 3 uses ``inspect.signature``, which
    works for many more types of callables.

    Many builtins in the standard library are also supported.
    """

from . import _signatures as _sigs
