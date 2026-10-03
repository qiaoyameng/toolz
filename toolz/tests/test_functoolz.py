import functools
import inspect
import sys
import typing
import pytest
import toolz
from toolz.functoolz import (thread_first, thread_last, memoize, curry,
                             compose, compose_left, pipe, complement, do, juxt,
                             flip, excepts, apply, Compose, placeholder)
from operator import add, mul, itemgetter
from toolz.utils import raises
from functools import partial


def iseven(x):
    return x % 2 == 0


def isodd(x):
    return x % 2 == 1


def inc(x):
    return x + 1


def double(x):
    return 2 * x


class AlwaysEquals:
    """useful to test correct __eq__ implementation of other objects"""

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


class NeverEquals:
    """useful to test correct __eq__ implementation of other objects"""

    def __eq__(self, other):
        return False

    def __ne__(self, other):
        return True


def test_apply():
    assert apply(double, 5) == 10
    assert tuple(map(apply, [double, inc, double], [10, 500, 8000])) == (20, 501, 16000)
    assert raises(TypeError, apply)


def test_thread_first():
    assert thread_first(2) == 2
    assert thread_first(2, inc) == 3
    assert thread_first(2, inc, inc) == 4
    assert thread_first(2, double, inc) == 5
    assert thread_first(2, (add, 5), double) == 14


def test_thread_last():
    assert list(thread_last([1, 2, 3], (map, inc), (filter, iseven))) == [2, 4]
    assert list(thread_last([1, 2, 3], (map, inc), (filter, isodd))) == [3]
    assert thread_last(2, (add, 5), double) == 14


def test_memoize():
    fn_calls = [0]  # Storage for side effects

    def f(x, y):
        """ A docstring """
        fn_calls[0] += 1
        return x + y
    mf = memoize(f)

    assert mf(2, 3) is mf(2, 3)
    assert fn_calls == [1]  # function was only called once
    assert mf.__doc__ == f.__doc__
    assert raises(TypeError, lambda: mf(1, {}))


def test_memoize_kwargs():
    fn_calls = [0]  # Storage for side effects

    def f(x, y=0):
        return x + y

    mf = memoize(f)

    assert mf(1) == f(1)
    assert mf(1, 2) == f(1, 2)
    assert mf(1, y=2) == f(1, y=2)
    assert mf(1, y=3) == f(1, y=3)


def test_memoize_curried():
    @curry
    def f(x, y=0):
        return x + y

    f2 = f(y=1)
    fm2 = memoize(f2)

    assert fm2(3) == f2(3)
    assert fm2(3) == f2(3)


def test_memoize_partial():
    def f(x, y=0):
        return x + y

    f2 = partial(f, y=1)
    fm2 = memoize(f2)

    assert fm2(3) == f2(3)
    assert fm2(3) == f2(3)


def test_memoize_key_signature():
    # Single argument should not be tupled as a key.  No keywords.
    mf = memoize(lambda x: False, cache={1: True})
    assert mf(1) is True
    assert mf(2) is False

    # Single argument must be tupled if signature has varargs.  No keywords.
    mf = memoize(lambda x, *args: False, cache={(1,): True, (1, 2): 2})
    assert mf(1) is True
    assert mf(2) is False
    assert mf(1, 1) is False
    assert mf(1, 2) == 2
    assert mf((1, 2)) is False

    # More than one argument is always tupled.  No keywords.
    mf = memoize(lambda x, y: False, cache={(1, 2): True})
    assert mf(1, 2) is True
    assert mf(1, 3) is False
    assert raises(TypeError, lambda: mf((1, 2)))

    # Nullary function (no inputs) uses empty tuple as the key
    mf = memoize(lambda: False, cache={(): True})
    assert mf() is True

    # Single argument must be tupled if there are keyword arguments, because
    # keyword arguments may be passed as unnamed args.
    mf = memoize(lambda x, y=0: False,
                 cache={((1,), frozenset((('y', 2),))): 2,
                        ((1, 2), None): 3})
    assert mf(1, y=2) == 2
    assert mf(1, 2) == 3
    assert mf(2, y=2) is False
    assert mf(2, 2) is False
    assert mf(1) is False
    assert mf((1, 2)) is False

    # Keyword-only signatures must still have an "args" tuple.
    mf = memoize(lambda x=0: False, cache={(None, frozenset((('x', 1),))): 1,
                                           ((1,), None): 2})
    assert mf() is False
    assert mf(x=1) == 1
    assert mf(1) == 2


def test_memoize_curry_cache():
    @memoize(cache={1: True})
    def f(x):
        return False

    assert f(1) is True
    assert f(2) is False


def test_memoize_key():
    @memoize(key=lambda args, kwargs: args[0])
    def f(x, y, *args, **kwargs):
        return x + y

    assert f(1, 2) == 3
    assert f(1, 3) == 3


def test_memoize_wrapped():

    def foo():
        """
        Docstring
        """
        pass
    memoized_foo = memoize(foo)
    assert memoized_foo.__wrapped__ is foo


def test_curry_simple():
    cmul = curry(mul)
    double = cmul(2)
    assert callable(double)
    assert double(10) == 20
    assert repr(cmul) == repr(mul)

    cmap = curry(map)
    assert list(cmap(inc)([1, 2, 3])) == [2, 3, 4]

    assert raises(TypeError, lambda: curry())
    assert raises(TypeError, lambda: curry({1: 2}))


def test_curry_kwargs():
    def f(a, b, c=10):
        return (a + b) * c

    f = curry(f)
    assert f(1, 2, 3) == 9
    assert f(1)(2, 3) == 9
    assert f(1, 2) == 30
    assert f(1, c=3)(2) == 9
    assert f(c=3)(1, 2) == 9

    def g(a=1, b=10, c=0):
        return a + b + c

    cg = curry(g, b=2)
    assert cg() == 3
    assert cg(b=3) == 4
    assert cg(a=0) == 2
    assert cg(a=0, b=1) == 1
    assert cg(0) == 2  # pass "a" as arg, not kwarg
    assert raises(TypeError, lambda: cg(1, 2))  # pass "b" as arg AND kwarg

    def h(x, func=int):
        return func(x)

    # __init__ must not pick func as positional arg
    assert curry(h)(0.0) == 0
    assert curry(h)(func=str)(0.0) == '0.0'
    assert curry(h, func=str)(0.0) == '0.0'


def test_curry_passes_errors():
    @curry
    def f(a, b):
        if not isinstance(a, int):
            raise TypeError()
        return a + b

    assert f(1, 2) == 3
    assert raises(TypeError, lambda: f('1', 2))
    assert raises(TypeError, lambda: f('1')(2))
    assert raises(TypeError, lambda: f(1, 2, 3))


def test_curry_docstring():
    def f(x, y):
        """ A docstring """
        return x

    g = curry(f)
    assert g.__doc__ == f.__doc__
    assert str(g) == str(f)
    assert f(1, 2) == g(1, 2)


def test_curry_is_like_partial():
    def foo(a, b, c=1):
        return a + b + c

    p, c = partial(foo, 1, c=2), curry(foo)(1, c=2)
    assert p.keywords == c.keywords
    assert p.args == c.args
    assert p(3) == c(3)

    p, c = partial(foo, 1), curry(foo)(1)
    assert p.keywords == c.keywords
    assert p.args == c.args
    assert p(3) == c(3)
    assert p(3, c=2) == c(3, c=2)

    p, c = partial(foo, c=1), curry(foo)(c=1)
    assert p.keywords == c.keywords
    assert p.args == c.args
    assert p(1, 2) == c(1, 2)


def test_curry_is_idempotent():
    def foo(a, b, c=1):
        return a + b + c

    f = curry(foo, 1, c=2)
    g = curry(f)
    assert isinstance(f, curry)
    assert isinstance(g, curry)
    assert not isinstance(g.func, curry)
    assert not hasattr(g.func, 'func')
    assert f.func == g.func
    assert f.args == g.args
    assert f.keywords == g.keywords


def test_curry_attributes_readonly():
    def foo(a, b, c=1):
        return a + b + c

    f = curry(foo, 1, c=2)
    assert raises(AttributeError, lambda: setattr(f, 'args', (2,)))
    assert raises(AttributeError, lambda: setattr(f, 'keywords', {'c': 3}))
    assert raises(AttributeError, lambda: setattr(f, 'func', f))
    assert raises(AttributeError, lambda: delattr(f, 'args'))
    assert raises(AttributeError, lambda: delattr(f, 'keywords'))
    assert raises(AttributeError, lambda: delattr(f, 'func'))


def test_curry_attributes_writable():
    def foo(a, b, c=1):
        return a + b + c
    foo.__qualname__ = 'this.is.foo'
    f = curry(foo, 1, c=2)
    assert f.__qualname__ == 'this.is.foo'
    f.__name__ = 'newname'
    f.__doc__ = 'newdoc'
    f.__module__ = 'newmodule'
    f.__qualname__ = 'newqualname'
    assert f.__name__ == 'newname'
    assert f.__doc__ == 'newdoc'
    assert f.__module__ == 'newmodule'
    assert f.__qualname__ == 'newqualname'
    if hasattr(f, 'func_name'):
        assert f.__name__ == f.func_name


def test_curry_module():
    from toolz.curried.exceptions import merge
    assert merge.__module__ == 'toolz.curried.exceptions'


def test_curry_comparable():
    def foo(a, b, c=1):
        return a + b + c
    f1 = curry(foo, 1, c=2)
    f2 = curry(foo, 1, c=2)
    g1 = curry(foo, 1, c=3)
    h1 = curry(foo, c=2)
    h2 = h1(c=2)
    h3 = h1()
    assert f1 == f2
    assert not (f1 != f2)
    assert f1 != g1
    assert not (f1 == g1)
    assert f1 != h1
    assert h1 == h2
    assert h1 == h3

    # test function comparison works
    def bar(a, b, c=1):
        return a + b + c
    b1 = curry(bar, 1, c=2)
    assert b1 != f1

    assert {f1, f2, g1, h1, h2, h3, b1, b1()} == {f1, g1, h1, b1}

    # test unhashable input
    unhash1 = curry(foo, [])
    assert raises(TypeError, lambda: hash(unhash1))
    unhash2 = curry(foo, c=[])
    assert raises(TypeError, lambda: hash(unhash2))


def test_curry_doesnot_transmogrify():
    # Early versions of `curry` transmogrified to `partial` objects if
    # only one positional argument remained even if keyword arguments
    # were present.  Now, `curry` should always remain `curry`.
    def f(x, y=0):
        return x + y

    cf = curry(f)
    assert cf(y=1)(y=2)(y=3)(1) == f(1, 3)


def test_curry_on_classmethods():
    class A:
        BASE = 10

        def __init__(self, base):
            self.BASE = base

        @curry
        def addmethod(self, x, y):
            return self.BASE + x + y

        @classmethod
        @curry
        def addclass(cls, x, y):
            return cls.BASE + x + y

        @staticmethod
        @curry
        def addstatic(x, y):
            return x + y

    a = A(100)
    assert a.addmethod(3, 4) == 107
    assert a.addmethod(3)(4) == 107
    assert A.addmethod(a, 3, 4) == 107
    assert A.addmethod(a)(3)(4) == 107

    assert a.addclass(3, 4) == 17
    assert a.addclass(3)(4) == 17
    assert A.addclass(3, 4) == 17
    assert A.addclass(3)(4) == 17

    assert a.addstatic(3, 4) == 7
    assert a.addstatic(3)(4) == 7
    assert A.addstatic(3, 4) == 7
    assert A.addstatic(3)(4) == 7

    # we want this to be of type curry
    assert isinstance(a.addmethod, curry)
    assert isinstance(A.addmethod, curry)


def test_memoize_on_classmethods():
    class A:
        BASE = 10
        HASH = 10

        def __init__(self, base):
            self.BASE = base

        @memoize
        def addmethod(self, x, y):
            return self.BASE + x + y

        @classmethod
        @memoize
        def addclass(cls, x, y):
            return cls.BASE + x + y

        @staticmethod
        @memoize
        def addstatic(x, y):
            return x + y

        def __hash__(self):
            return self.HASH

    a = A(100)
    assert a.addmethod(3, 4) == 107
    assert A.addmethod(a, 3, 4) == 107

    a.BASE = 200
    assert a.addmethod(3, 4) == 107
    a.HASH = 200
    assert a.addmethod(3, 4) == 207

    assert a.addclass(3, 4) == 17
    assert A.addclass(3, 4) == 17
    A.BASE = 20
    assert A.addclass(3, 4) == 17
    A.HASH = 20  # hashing of class is handled by metaclass
    assert A.addclass(3, 4) == 17  # hence, != 27

    assert a.addstatic(3, 4) == 7
    assert A.addstatic(3, 4) == 7


def test_curry_call():
    @curry
    def add(x, y):
        return x + y
    assert raises(TypeError, lambda: add.call(1))
    assert add(1)(2) == add.call(1, 2)
    assert add(1)(2) == add(1).call(2)


def test_curry_bind():
    @curry
    def add(x=1, y=2):
        return x + y
    assert add() == add(1, 2)
    assert add.bind(10)(20) == add(10, 20)
    assert add.bind(10).bind(20)() == add(10, 20)
    assert add.bind(x=10)(y=20) == add(10, 20)
    assert add.bind(x=10).bind(y=20)() == add(10, 20)


def test_curry_unknown_args():
    def add3(x, y, z):
        return x + y + z

    @curry
    def f(*args):
        return add3(*args)

    assert f()(1)(2)(3) == 6
    assert f(1)(2)(3) == 6
    assert f(1, 2)(3) == 6
    assert f(1, 2, 3) == 6
    assert f(1, 2)(3, 4) == f(1, 2, 3, 4)


def test_curry_placeholder_simple():
    def f(a, b, c, d):
        return a, b, c, d

    cf = curry(f)
    # Fix the second and fourth positional arguments; fill the rest later
    g = cf(placeholder, 2, placeholder, 4)
    assert isinstance(g, curry)
    assert g(1, 3) == (1, 2, 3, 4)
    # Extra positional arguments are appended after open slots are filled
    assert cf(placeholder, 2)(1, 3, 4) == (1, 2, 3, 4)
    # Placeholders may be introduced across multiple calls
    step = cf(placeholder)(1, placeholder)(2, placeholder)(3, placeholder)
    assert isinstance(step, curry)
    assert step(4) == (1, 2, 3, 4)
    # Placeholders given at construction time work too
    assert curry(f, placeholder, 2)(1, 3, 4) == (1, 2, 3, 4)
    # The placeholder is a singleton, also exposed on curry and toolz
    assert curry.placeholder is placeholder
    assert toolz.placeholder is placeholder
    assert repr(placeholder) == 'placeholder'


def test_curry_placeholder_kwargs():
    def f(a, b, c=10):
        return a, b, c

    cf = curry(f)
    assert cf(placeholder, c=3)(1, 2) == (1, 2, 3)
    assert cf(placeholder, 2)(1, c=3) == (1, 2, 3)
    # Keyword arguments bind per the Python signature; a parameter may not
    # be occupied both positionally and by keyword
    assert raises(TypeError, lambda: cf(1, placeholder)(b=2))
    assert raises(TypeError, lambda: cf(placeholder, 2)(1, b=5))


def test_curry_placeholder_posonly_kwonly():
    def f(a, b, /, c, *, d):
        return a, b, c, d

    cf = curry(f)
    assert cf(placeholder, 2)(1, placeholder, d=4)(3) == (1, 2, 3, 4)
    assert cf(1, placeholder)(2, 3, d=4) == (1, 2, 3, 4)
    # Positional-only parameters cannot be filled by keyword
    assert raises(TypeError, lambda: cf(placeholder, 2)(1, 3, b=5))


def test_curry_placeholder_varargs():
    def f(*args):
        return args

    cf = curry(f)
    assert cf(placeholder, 2)(1) == (1, 2)
    assert cf(placeholder, 2)(1, 3, 4) == (1, 2, 3, 4)

    def add3(x, y, z):
        return x + y + z

    @curry
    def g(*args):
        return add3(*args)

    assert g(placeholder, 2)(1, 3) == 6


def test_curry_placeholder_defaults():
    def f(a, b=2, c=3):
        return a, b, c

    cf = curry(f)
    assert cf(placeholder)(1) == (1, 2, 3)
    assert cf(placeholder, c=30)(1) == (1, 2, 30)
    assert cf(placeholder, 20)(1) == (1, 20, 3)


def test_curry_placeholder_method():
    class A:
        def __init__(self, x):
            self.x = x

        @curry
        def method(self, a, b, c):
            return self.x, a, b, c

    a = A(0)
    assert a.method(placeholder, 2)(1, 3) == (0, 1, 2, 3)
    assert A.method(a, placeholder, 2)(1, 3) == (0, 1, 2, 3)


def test_curry_placeholder_nested():
    def f(a, b, c, d):
        return a, b, c, d

    inner = curry(f)(1, placeholder)
    assert curry(inner)(2, placeholder, 4)(3) == (1, 2, 3, 4)
    assert curry(inner)(placeholder, 3)(2, 4) == (1, 2, 3, 4)
    # Filling all slots may still leave a normally-curried call
    def g(a, b, c):
        return a, b, c
    assert curry(g)(1, placeholder)(2)(3) == (1, 2, 3)


def test_curry_placeholder_signature():
    def f(a, b, c):
        return a + b + c

    sig = inspect.signature(curry(f)(1, placeholder, 3))
    assert list(sig.parameters) == ['b']

    def g(a, b=2):
        return a + b

    sig = inspect.signature(curry(g)(placeholder))
    assert list(sig.parameters) == ['a', 'b']
    assert sig.parameters['b'].default == 2

    def h(a, b, /, c):
        return a + b + c

    sig = inspect.signature(curry(h)(1, placeholder))
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ['b', 'c']
    assert params[0].kind == params[0].POSITIONAL_ONLY


def test_curry_placeholder_typeerror_in_body():
    @curry
    def f(a, b):
        if not isinstance(a, int):
            raise TypeError('a must be an int')
        return a + b

    # The function is not called while placeholders remain unfilled
    g = f(placeholder, 2)
    assert isinstance(g, curry)
    assert g(1) == 3
    # A genuine TypeError raised by the function body is not mistaken
    # for an incomplete call
    assert raises(TypeError, lambda: g('1'))
    assert raises(TypeError, lambda: f(placeholder, 2)('1', 3))


def test_curry_placeholder_call():
    @curry
    def add(x, y):
        return x + y

    g = add(placeholder, 2)
    assert raises(TypeError, lambda: g.call())
    assert raises(TypeError, lambda: g.call(placeholder))
    assert raises(TypeError, lambda: add.call(placeholder, 2))
    assert g.call(1) == 3


def test_curry_placeholder_bind():
    @curry
    def add3(x, y, z):
        return x + y + z

    g = add3.bind(placeholder, 2)
    assert isinstance(g, curry)
    assert g(1, 3) == 6
    assert add3(placeholder, 2).bind(1)(3) == 6


def test_curry_placeholder_comparable():
    def f(a, b, c):
        return a, b, c

    assert curry(f)(1, placeholder) == curry(f)(1, placeholder)
    assert curry(f)(1, placeholder) != curry(f)(placeholder, 1)
    assert hash(curry(f)(1, placeholder)) == hash(curry(f)(1, placeholder))


def test_curry_bad_types():
    assert raises(TypeError, lambda: curry(1))


def test_curry_subclassable():
    class mycurry(curry):
        pass

    add = mycurry(lambda x, y: x+y)
    assert isinstance(add, curry)
    assert isinstance(add, mycurry)
    assert isinstance(add(1), mycurry)
    assert isinstance(add()(1), mycurry)
    assert add(1)(2) == 3

    # Should we make `_should_curry` public?
    """
    class curry2(curry):
        def _should_curry(self, args, kwargs, exc=None):
            return len(self.args) + len(args) < 2

    add = curry2(lambda x, y: x+y)
    assert isinstance(add(1), curry2)
    assert add(1)(2) == 3
    assert isinstance(add(1)(x=2), curry2)
    assert raises(TypeError, lambda: add(1)(x=2)(3))
    """


def generate_compose_test_cases():
    """
    Generate test cases for parametrized tests of the compose function.
    """

    def add_then_multiply(a, b, c=10):
        return (a + b) * c

    return (
        (
            (),        # arguments to compose()
            (0,), {},  # positional and keyword args to the Composed object
            0          # expected result
        ),
        (
            (inc,),
            (0,), {},
            1
        ),
        (
            (double, inc),
            (0,), {},
            2
        ),
        (
            (str, iseven, inc, double),
            (3,), {},
            "False"
        ),
        (
            (str, add),
            (1, 2), {},
            '3'
        ),
        (
            (str, inc, add_then_multiply),
            (1, 2), {"c": 3},
            '10'
        ),
    )


def test_compose():
    for (compose_args, args, kw, expected) in generate_compose_test_cases():
        assert compose(*compose_args)(*args, **kw) == expected


def test_compose_metadata():

    # Define two functions with different names
    def f(a):
        return a

    def g(a):
        return a

    composed = compose(f, g)
    assert composed.__name__ == 'f_of_g'
    assert composed.__doc__ == 'lambda *args, **kwargs: f(g(*args, **kwargs))'

    # Create an object with no __name__.
    h = object()

    composed = compose(f, h)
    assert composed.__name__ == 'Compose'
    assert composed.__doc__ == 'A composition of functions'

    assert repr(composed) == f'Compose({f!r}, {h!r})'

    assert composed == compose(f, h)
    assert composed == AlwaysEquals()
    assert not composed == compose(h, f)
    assert not composed == object()
    assert not composed == NeverEquals()

    assert composed != compose(h, f)
    assert composed != NeverEquals()
    assert composed != object()
    assert not composed != compose(f, h)
    assert not composed != AlwaysEquals()

    assert hash(composed) == hash(compose(f, h))
    assert hash(composed) != hash(compose(h, f))

    bindable = compose(str, lambda x: x*2, lambda x, y=0: int(x) + y)

    class MyClass:

        def __int__(self):
            return 8

        my_method = bindable
        my_static_method = staticmethod(bindable)

    assert MyClass.my_method(3) == '6'
    assert MyClass.my_method(3, y=2) == '10'
    assert MyClass.my_static_method(2) == '4'
    assert MyClass().my_method() == '16'
    assert MyClass().my_method(y=3) == '22'
    assert MyClass().my_static_method(0) == '0'
    assert MyClass().my_static_method(0, 1) == '2'

    assert compose(f, h).__wrapped__ is h
    if hasattr(toolz, 'sandbox'):  # only test this with Python version (i.e., not Cython)
        assert compose(f, h).__class__.__wrapped__ is None

    # __signature__ is python3 only

    def myfunc(a, b, c, *d, **e):
        return 4

    def otherfunc(f):
        return f'result: {f}'

    # set annotations compatibly with python2 syntax
    myfunc.__annotations__ = {
        'a': int,
        'b': str,
        'c': float,
        'd': int,
        'e': bool,
        'return': int,
    }
    otherfunc.__annotations__ = {'f': int, 'return': str}

    composed = compose(otherfunc, myfunc)
    sig = inspect.signature(composed)
    assert sig.parameters == inspect.signature(myfunc).parameters
    assert sig.return_annotation == str

    class MyClass:
        method = composed

    assert len(inspect.signature(MyClass().method).parameters) == 4


def test_compose_annotations():
    # Parameter annotations come from the first function applied and the
    # return annotation from the last one: the same view as __signature__.
    def f(x: int) -> str:
        return str(x)

    def g(y: str) -> float:
        return float(len(y))

    def h(z):
        return z

    class A:
        attr: bytes  # class annotations are not part of the signature

        def __init__(self, a: int):
            pass

        def method(self, m: int) -> str:
            return str(m)

    assert compose(g, f).__annotations__ == {'x': int, 'return': float}
    assert compose(h, f).__annotations__ == {'x': int}
    assert compose(f, h).__annotations__ == {'return': str}
    assert compose(h, h).__annotations__ == {}
    assert compose(h, g, f).__annotations__ == {'x': int}
    assert compose(g, f, h).__annotations__ == {'return': float}
    # callables without a signature contribute nothing
    assert compose(str, object()).__annotations__ == {}
    assert compose(str, f).__annotations__ == {'x': int}
    assert compose(g, len).__annotations__ == {'return': float}

    # consistent with the synthesized signature
    for c in [compose(g, f), compose(g, partial(f)), compose(g, curry(f)),
              compose(g, A(0).method), compose(g, A), compose(f, A),
              compose(compose(g, f), compose(f, h))]:
        sig = inspect.signature(c)
        expected = {k: p.annotation for k, p in sig.parameters.items()
                    if p.annotation is not p.empty}
        if sig.return_annotation is not sig.empty:
            expected['return'] = sig.return_annotation
        assert c.__annotations__ == expected
        assert typing.get_type_hints(c) == expected

    assert compose(g, A).__annotations__ == {'a': int, 'return': float}
    assert compose(g, A(0).method).__annotations__ == {
        'm': int, 'return': float}

    # the class itself still has ordinary (empty) annotations
    assert typing.get_type_hints(Compose) == {}
    if hasattr(inspect, 'get_annotations'):  # Python >= 3.10
        assert inspect.get_annotations(Compose) == {}

    # functools.wraps copies the annotations
    def wrapper(*args, **kwargs):
        pass

    wrapped = functools.wraps(compose(g, f))(wrapper)
    assert wrapped.__annotations__ == {'x': int, 'return': float}

    # string annotations (PEP 563) are kept as-is and resolved by typing
    ns = {}
    exec('from __future__ import annotations\n'
         'def s(x: int) -> str: return str(x)', ns)
    assert compose(g, ns['s']).__annotations__ == {'x': 'int', 'return': float}
    assert compose(ns['s'], f).__annotations__ == {'x': int, 'return': 'str'}
    assert typing.get_type_hints(compose(g, ns['s'])) == {'x': int,
                                                          'return': float}


@pytest.mark.skipif(sys.version_info < (3, 14),
                    reason='annotationlib is new in Python 3.14')
def test_compose_annotations_formats():
    import annotationlib
    Format = annotationlib.Format

    ns = {}
    # PEP 649: annotations are evaluated lazily, so this may be undefined
    exec('def f(x: Undefined) -> int: return 1', ns)
    f = ns['f']

    def g(y: int) -> str:
        return str(y)

    c = compose(g, f)
    ref = annotationlib.get_annotations(f, format=Format.FORWARDREF)['x']
    assert annotationlib.get_annotations(c, format=Format.FORWARDREF) == {
        'x': ref, 'return': str}
    assert annotationlib.get_annotations(c, format=Format.STRING) == {
        'x': 'Undefined', 'return': 'str'}
    assert raises(NameError, lambda: c.__annotations__)
    assert annotationlib.get_annotations(compose(g, c),
                                         format=Format.FORWARDREF) == {
        'x': ref, 'return': str}
    # curry.__signature__ ignores the format; still must not raise NameError
    cc = compose(g, curry(f))
    assert raises(NameError, lambda: cc.__annotations__)
    assert annotationlib.get_annotations(cc, format=Format.FORWARDREF) == {
        'return': str}
    assert annotationlib.get_annotations(cc, format=Format.STRING) == {
        'return': 'str'}
    # ... and STRING gives strings even though curry's signature has values
    assert annotationlib.get_annotations(compose(g, curry(g)),
                                         format=Format.STRING) == {
        'y': 'int', 'return': 'str'}

    # PEP 649: unsupported formats must raise NotImplementedError
    assert compose(g, g).__annotate__(Format.VALUE) == {
        'y': int, 'return': str}
    assert raises(NotImplementedError,
                  lambda: c.__annotate__(Format.VALUE_WITH_FAKE_GLOBALS))
    assert raises(NotImplementedError, lambda: c.__annotate__(99))

    # functools.wraps copies __annotate__ on 3.14+
    wrapped = functools.wraps(c)(lambda *args, **kwargs: None)
    assert annotationlib.get_annotations(wrapped, format=Format.STRING) == {
        'x': 'Undefined', 'return': 'str'}

    assert annotationlib.get_annotations(Compose) == {}
    assert annotationlib.get_annotations(Compose, format=Format.STRING) == {}


def generate_compose_left_test_cases():
    """
    Generate test cases for parametrized tests of the compose function.

    These are based on, and equivalent to, those produced by
    enerate_compose_test_cases().
    """
    return tuple(
        (tuple(reversed(compose_args)), args, kwargs, expected)
        for (compose_args, args, kwargs, expected)
        in generate_compose_test_cases()
    )


def test_compose_left():
    for (compose_left_args, args, kw, expected) in generate_compose_left_test_cases():
        assert compose_left(*compose_left_args)(*args, **kw) == expected


def test_pipe():
    assert pipe(1, inc) == 2
    assert pipe(1, inc, inc) == 3
    assert pipe(1, double, inc, iseven) is False


def test_complement():
    # No args:
    assert complement(lambda: False)()
    assert not complement(lambda: True)()

    # Single arity:
    assert complement(iseven)(1)
    assert not complement(iseven)(2)
    assert complement(complement(iseven))(2)
    assert not complement(complement(isodd))(2)

    # Multiple arities:
    both_even = lambda a, b: iseven(a) and iseven(b)
    assert complement(both_even)(1, 2)
    assert not complement(both_even)(2, 2)

    # Generic truthiness:
    assert complement(lambda: "")()
    assert complement(lambda: 0)()
    assert complement(lambda: None)()
    assert complement(lambda: [])()

    assert not complement(lambda: "x")()
    assert not complement(lambda: 1)()
    assert not complement(lambda: [1])()


def test_do():
    inc = lambda x: x + 1
    assert do(inc, 1) == 1

    log = []
    assert do(log.append, 1) == 1
    assert log == [1]


def test_juxt_generator_input():
    data = list(range(10))
    juxtfunc = juxt(itemgetter(2*i) for i in range(5))
    assert juxtfunc(data) == (0, 2, 4, 6, 8)
    assert juxtfunc(data) == (0, 2, 4, 6, 8)


def test_flip():
    def f(a, b):
        return a, b

    assert flip(f, 'a', 'b') == ('b', 'a')


def test_excepts():
    # These are descriptors, make sure this works correctly.
    assert excepts.__name__ == 'excepts'
    # in Python < 3.13 the second line is indented, in 3.13+
    # it is not, strip all lines to fudge it
    testlines = "\n".join(line.strip() for line in excepts.__doc__.splitlines())
    assert (
        'A wrapper around a function to catch exceptions and\n'
        'dispatch to a handler.\n'
    ) in testlines

    def idx(a):
        """idx docstring
        """
        return [1, 2].index(a)

    def handler(e):
        """handler docstring
        """
        assert isinstance(e, ValueError)
        return -1

    excepting = excepts(ValueError, idx, handler)
    assert excepting(1) == 0
    assert excepting(2) == 1
    assert excepting(3) == -1

    assert excepting.__name__ == 'idx_excepting_ValueError'
    assert 'idx docstring' in excepting.__doc__
    assert 'ValueError' in excepting.__doc__
    assert 'handler docstring' in excepting.__doc__

    def getzero(a):
        """getzero docstring
        """
        return a[0]

    excepting = excepts((IndexError, KeyError), getzero)
    assert excepting([]) is None
    assert excepting([1]) == 1
    assert excepting({}) is None
    assert excepting({0: 1}) == 1

    assert excepting.__name__ == 'getzero_excepting_IndexError_or_KeyError'
    assert 'getzero docstring' in excepting.__doc__
    assert 'return_none' in excepting.__doc__
    assert 'Returns None' in excepting.__doc__

    def raise_(a):
        """A function that raises an instance of the exception type given.
        """
        raise a()

    excepting = excepts((ValueError, KeyError), raise_)
    assert excepting(ValueError) is None
    assert excepting(KeyError) is None
    assert raises(TypeError, lambda: excepting(TypeError))
    assert raises(NotImplementedError, lambda: excepting(NotImplementedError))

    excepting = excepts(object(), object(), object())
    assert excepting.__name__ == 'excepting'
    assert excepting.__doc__ == excepts.__doc__
