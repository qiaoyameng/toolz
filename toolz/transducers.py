""" Reducing transforms: map, filter, partition and take over a reducer

This module provides a small transducer-style interface.  Instead of
wrapping each stage of a pipeline in its own iterator, transforms are
composed once and then attached directly to a reducer, so every input
item flows straight into the reduction.

A *reducer* is anything with a ``step(state, item)`` method; the
``transduce`` driver also accepts plain callables.  A step may return
``reduced(state)`` to signal early completion, in which case no further
items are pulled from the input.  An optional ``complete`` callback is
invoked exactly once per run, on both natural exhaustion and early
termination.  Exceptions raised by ``step`` or ``complete`` propagate
unchanged.

Composed transforms are re-usable across inputs: all buffering state
lives in the per-run reducer chain, never on the transform itself.

>>> from toolz.transducers import compose, mapping, filtering, transduce
>>> xf = compose(mapping(lambda x: x + 1), filtering(lambda x: x % 2 == 0))
>>> def collect(state, item):
...     state.append(item)
...     return state
>>> transduce(xf, collect, [], [1, 2, 3, 4])
[2, 4]
>>> transduce(xf, collect, [], [10, 11])  # same transform, fresh state
[12]
"""
__all__ = ('Reduced', 'reduced', 'is_reduced', 'XForm', 'mapping',
           'filtering', 'partitioning', 'taking', 'compose', 'transduce')


class Reduced:
    """ Wrapper around a state signalling early completion of a reduction

    Return ``Reduced(state)`` from a step function to stop the reduction
    immediately; no further items are pulled from the input.
    """
    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f'Reduced({self.value!r})'


def reduced(value):
    """ Wrap ``value`` to signal that a reduction should stop early

    >>> from toolz.transducers import compose, transduce, reduced
    >>> def step(state, item):
    ...     if item > 2:
    ...         return reduced(state)
    ...     return state + [item]
    >>> transduce(compose(), step, [], [1, 2, 3, 4])
    [1, 2]
    """
    return Reduced(value)


def is_reduced(value):
    """ Return True if ``value`` is wrapped in ``Reduced`` """
    return isinstance(value, Reduced)


class _CallableReducer:
    """ Adapts user-supplied ``step``/``complete`` callables to a reducer """
    def __init__(self, step, complete):
        self._step = step
        self._complete = complete

    def step(self, state, item):
        return self._step(state, item)

    def complete(self, state, terminated):
        if self._complete is None:
            return state
        return self._complete(state)


class _Map:
    def __init__(self, func, reducer):
        self._func = func
        self._reducer = reducer

    def step(self, state, item):
        return self._reducer.step(state, self._func(item))

    def complete(self, state, terminated):
        return self._reducer.complete(state, terminated)


class _Filter:
    def __init__(self, predicate, reducer):
        self._predicate = predicate
        self._reducer = reducer

    def step(self, state, item):
        if self._predicate(item):
            return self._reducer.step(state, item)
        return state

    def complete(self, state, terminated):
        return self._reducer.complete(state, terminated)


class _Take:
    def __init__(self, n, reducer):
        self._remaining = n
        self._reducer = reducer

    def done(self):
        return self._remaining <= 0

    def step(self, state, item):
        self._remaining -= 1
        result = self._reducer.step(state, item)
        if self._remaining <= 0 and not is_reduced(result):
            return Reduced(result)
        return result

    def complete(self, state, terminated):
        return self._reducer.complete(state, terminated)


class _Partition:
    def __init__(self, n, reducer):
        self._n = n
        self._reducer = reducer
        self._buffer = []

    def step(self, state, item):
        self._buffer.append(item)
        if len(self._buffer) == self._n:
            batch = tuple(self._buffer)
            del self._buffer[:]
            return self._reducer.step(state, batch)
        return state

    def complete(self, state, terminated):
        buffer, self._buffer = self._buffer, []
        if buffer and not terminated:
            # Natural end: emit the partial tail batch.  On early
            # termination the boundary already delivered downstream
            # stands; the leftover partial batch is discarded.
            state = self._reducer.step(state, tuple(buffer))
            if is_reduced(state):
                state = state.value
        return self._reducer.complete(state, terminated)


class XForm:
    """ A composable reducing transform

    An ``XForm`` turns a reducer into a new reducer.  Instances are
    created with ``mapping``, ``filtering``, ``partitioning`` and
    ``taking`` and combined with ``compose``.  They are immutable and
    may be re-used for any number of ``transduce`` runs.
    """
    __slots__ = ('_build',)

    def __init__(self, build):
        self._build = build

    def __call__(self, reducer):
        """ Attach this transform to ``reducer``, returning a new reducer """
        return self._build(reducer)

    def then(self, *xforms):
        """ Compose so that data flows through this transform first

        >>> from toolz.transducers import mapping, taking
        >>> xf = mapping(str).then(taking(2))
        >>> xf.transduce(lambda s, x: s + [x], [], [1, 2, 3])
        ['1', '2']
        """
        return compose(self, *xforms)

    def transduce(self, step, init, iterable, complete=None):
        """ Reduce ``iterable`` through this transform

        Equivalent to ``transduce(self, step, init, iterable, complete)``.
        """
        return transduce(self, step, init, iterable, complete=complete)


def mapping(func):
    """ Transform that applies ``func`` to each item before reducing

    >>> from toolz.transducers import mapping
    >>> mapping(str).transduce(lambda s, x: s + [x], [], [1, 2, 3])
    ['1', '2', '3']
    """
    return XForm(lambda reducer: _Map(func, reducer))


def filtering(predicate):
    """ Transform that only passes items satisfying ``predicate``

    >>> from toolz.transducers import filtering
    >>> filtering(lambda x: x % 2 == 0).transduce(lambda s, x: s + [x], [], range(6))
    [0, 2, 4]
    """
    return XForm(lambda reducer: _Filter(predicate, reducer))


def partitioning(n):
    """ Transform that groups items into tuples of length at most ``n``

    On natural exhaustion the final partial batch is emitted, mirroring
    ``toolz.partition_all``.  On early termination the partial batch is
    discarded: the boundary already delivered downstream stands.

    >>> from toolz.transducers import partitioning
    >>> partitioning(2).transduce(lambda s, x: s + [x], [], [1, 2, 3])
    [(1, 2), (3,)]
    """
    if n < 1:
        raise ValueError('n must be at least 1')
    return XForm(lambda reducer: _Partition(n, reducer))


def taking(n):
    """ Transform that passes at most ``n`` items, then stops the reduction

    Once ``n`` items have been passed downstream no further items are
    pulled from the input.

    >>> from toolz.transducers import taking
    >>> taking(2).transduce(lambda s, x: s + [x], [], [1, 2, 3, 4])
    [1, 2]
    """
    if n < 0:
        raise ValueError('n must be non-negative')
    return XForm(lambda reducer: _Take(n, reducer))


def compose(*xforms):
    """ Compose transforms; data flows through the leftmost one first

    >>> from toolz.transducers import compose, mapping, taking
    >>> xf = compose(mapping(lambda x: x * 10), taking(2))
    >>> xf.transduce(lambda s, x: s + [x], [], [1, 2, 3])
    [10, 20]
    """
    if not xforms:
        return XForm(lambda reducer: reducer)

    def build(reducer):
        for xform in reversed(xforms):
            reducer = xform(reducer)
        return reducer
    return XForm(build)


def transduce(xform, step, init, iterable, complete=None):
    """ Reduce ``iterable`` with ``step``, transformed by ``xform``

    ``step(state, item)`` returns the new state, or ``reduced(state)``
    to stop early.  Reduction starts from ``init``.  If given,
    ``complete(state)`` is called exactly once at the end of the run --
    on natural exhaustion as well as on early termination -- and its
    return value is the result; otherwise the final state is returned.

    ``xform`` may be an ``XForm`` or any callable mapping a reducer to
    a reducer.  Exceptions raised by ``step`` or ``complete`` propagate
    unchanged.

    >>> from toolz.transducers import transduce, filtering
    >>> transduce(filtering(lambda x: x > 1), lambda s, x: s + x, 0, [1, 2, 3])
    5
    """
    reducer = xform(_CallableReducer(step, complete))
    state = init
    terminated = False
    iterator = iter(iterable)
    done = getattr(reducer, 'done', None)
    while True:
        if done is not None and done():
            terminated = True
            break
        try:
            item = next(iterator)
        except StopIteration:
            break
        state = reducer.step(state, item)
        if is_reduced(state):
            state = state.value
            terminated = True
            break
    return reducer.complete(state, terminated)
