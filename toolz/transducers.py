"""
Small reducer transformations.

The functions in this module connect transformations directly to a reducer.
Unlike iterator transformations, they do not create an intermediate iterator
for each stage.  Transformations are composed with :func:`toolz.compose` and
run with :func:`transduce`.
"""
import operator

from .functoolz import compose


__all__ = ('Reduced', 'is_reduced', 'reduced', 'transduce', 'map', 'filter',
           'partition', 'take', 'compose')


class Reduced:
    """A reducer result that signals early completion.

    >>> reduced_value = reduced(1)
    >>> is_reduced(reduced_value)
    True
    >>> reduced_value.value
    1
    """

    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f'{type(self).__name__}({self.value!r})'

    def __eq__(self, other):
        if isinstance(other, Reduced):
            return self.value == other.value
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result


def reduced(value):
    """Return ``value`` wrapped in :class:`Reduced`.

    Calling ``reduced`` on an already reduced value does not add another
    wrapper.

    >>> is_reduced(reduced(reduced(1)))
    True
    >>> reduced(reduced(1)).value
    1
    """
    if is_reduced(value):
        return value
    return Reduced(value)


def is_reduced(value):
    """Return whether ``value`` signals early completion."""
    return isinstance(value, Reduced)


class _Reducer:
    def start(self, state):
        return state

    def step(self, state, item):
        raise NotImplementedError

    def complete(self, state):
        return state


class _FinalReducer(_Reducer):
    __slots__ = ('_step', '_complete')

    def __init__(self, step, complete=None):
        self._step = step
        self._complete = complete

    def step(self, state, item):
        return self._step(state, item)

    def complete(self, state):
        if is_reduced(state):
            state = state.value
        if self._complete is None:
            return state
        return self._complete(state)


class _TransformReducer(_Reducer):
    __slots__ = ('reducer',)

    def __init__(self, reducer):
        self.reducer = reducer

    def start(self, state):
        return self.reducer.start(state)

    def complete(self, state):
        return self.reducer.complete(state)


class _MapReducer(_TransformReducer):
    __slots__ = ('func',)

    def __init__(self, reducer, func):
        super().__init__(reducer)
        self.func = func

    def step(self, state, item):
        if is_reduced(state):
            return state
        return self.reducer.step(state, self.func(item))


class _FilterReducer(_TransformReducer):
    __slots__ = ('predicate',)

    def __init__(self, reducer, predicate):
        super().__init__(reducer)
        self.predicate = predicate

    def step(self, state, item):
        if is_reduced(state):
            return state
        if self.predicate is None or self.predicate(item):
            return self.reducer.step(state, item)
        return state


class _TakeReducer(_TransformReducer):
    __slots__ = ('remaining',)

    def __init__(self, reducer, n):
        super().__init__(reducer)
        self.remaining = n

    def start(self, state):
        state = self.reducer.start(state)
        if self.remaining == 0:
            return reduced(state)
        return state

    def step(self, state, item):
        if is_reduced(state) or self.remaining == 0:
            return reduced(state)

        self.remaining -= 1
        result = self.reducer.step(state, item)
        if self.remaining == 0:
            return reduced(result)
        return result


class _PartitionReducer(_TransformReducer):
    __slots__ = ('n', 'buffer')

    def __init__(self, reducer, n):
        super().__init__(reducer)
        self.n = n
        self.buffer = []

    def step(self, state, item):
        if is_reduced(state):
            return state

        self.buffer.append(item)
        if len(self.buffer) == self.n:
            batch = tuple(self.buffer)
            self.buffer.clear()
            return self.reducer.step(state, batch)
        return state

    def complete(self, state):
        if not is_reduced(state) and self.buffer:
            batch = tuple(self.buffer)
            self.buffer.clear()
            state = self.reducer.step(state, batch)

        return self.reducer.complete(state)


def map(func):
    """Create a reducer transformation that applies ``func`` to each item.

    >>> from operator import add
    >>> transduce(map(lambda x: x + 1), add, 0, [1, 2, 3])
    9
    """
    def transducer(reducer):
        return _MapReducer(reducer, func)
    return transducer


def filter(predicate):
    """Create a reducer transformation that keeps selected items.

    >>> transduce(filter(lambda x: x % 2 == 0), lambda acc, x: acc + [x],
    ...           [], range(5))
    [0, 2, 4]
    """
    def transducer(reducer):
        return _FilterReducer(reducer, predicate)
    return transducer


def take(n):
    """Create a reducer transformation for the first ``n`` items.

    After ``n`` items have been passed downstream, the transformation returns
    a :class:`Reduced` value and ``transduce`` asks for no more upstream items.

    >>> transduce(take(2), lambda acc, x: acc + [x], [], range(5))
    [0, 1]
    """
    n = operator.index(n)
    if n < 0:
        raise ValueError('take requires a non-negative integer')

    def transducer(reducer):
        return _TakeReducer(reducer, n)
    return transducer


def partition(n):
    """Create a reducer transformation that groups items into tuples.

    A shorter final tuple is emitted when the input ends naturally.  If the
    pipeline completes early, a buffered partial tuple is retained rather than
    emitted past that boundary.

    >>> transduce(partition(2), lambda acc, x: acc + [x], [], range(5))
    [(0, 1), (2, 3), (4,)]
    """
    n = operator.index(n)
    if n < 1:
        raise ValueError('partition requires a positive integer')

    def transducer(reducer):
        return _PartitionReducer(reducer, n)
    return transducer


def transduce(transducer, step, init, coll, complete=None):
    """Reduce ``coll`` through a composed reducer transformation.

    Parameters
    ----------
    transducer : callable
        A transformation created by :func:`map`, :func:`filter`,
        :func:`partition`, or :func:`take`, composed together with
        :func:`compose`.
    step : callable
        The downstream reducer step, called as ``step(state, item)``.
    init
        The initial reducer state.
    coll : iterable
        The input collection.
    complete : callable, optional
        A completion callback, called as ``complete(state)`` once when the
        input is exhausted or the reducer completes early.

    >>> transduce(compose(map(lambda x: x + 1), filter(lambda x: x % 2)),
    ...           lambda acc, x: acc + [x], [], range(5))
    [1, 3, 5]
    """
    reducer = transducer(_FinalReducer(step, complete))
    state = reducer.start(init)

    if is_reduced(state):
        state = reducer.complete(state)
    else:
        for item in coll:
            state = reducer.step(state, item)
            if is_reduced(state):
                state = reducer.complete(state)
                break
        else:
            state = reducer.complete(state)

    if is_reduced(state):
        return state.value
    return state
