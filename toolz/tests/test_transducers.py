import pytest

from toolz import filter as tz_filter, map as tz_map, partition_all, take
from toolz.transducers import (
    Reduced,
    XForm,
    compose,
    filtering,
    is_reduced,
    mapping,
    partitioning,
    reduced,
    taking,
    transduce,
)


def collect(state, item):
    state.append(item)
    return state


class Logged:
    """Records pulls, steps and completions for a transduce run."""

    def __init__(self, items, stop_at=None):
        self.items = items
        self.stop_at = stop_at
        self.log = []

    def input(self):
        for item in self.items:
            self.log.append(('pull', item))
            yield item

    def step(self, state, item):
        self.log.append(('step', item))
        if self.stop_at is not None and len(state) + 1 >= self.stop_at:
            self.log.append(('reduce', item))
            return reduced(state + [item])
        return state + [item]

    def complete(self, state):
        self.log.append(('complete', tuple(state)))
        return state


def test_reduced_protocol():
    value = reduced(42)
    assert isinstance(value, Reduced)
    assert is_reduced(value)
    assert not is_reduced(42)
    assert value.value == 42
    assert repr(value) == 'Reduced(42)'


def test_mapping_matches_iterator_composition():
    xf = mapping(lambda x: x * 2)
    data = [1, 2, 3, 4]
    assert transduce(xf, collect, [], data) == list(tz_map(lambda x: x * 2, data))


def test_filtering_matches_iterator_composition():
    xf = filtering(lambda x: x % 2 == 0)
    data = range(10)
    assert transduce(xf, collect, [], data) == list(tz_filter(lambda x: x % 2 == 0, data))


def test_taking_matches_iterator_composition():
    xf = taking(3)
    data = [1, 2, 3, 4, 5]
    assert transduce(xf, collect, [], data) == list(take(3, data))


def test_composed_pipeline_matches_iterator_composition():
    def func(x):
        return x + 1

    def pred(x):
        return x % 2 == 0

    xf = compose(mapping(func), filtering(pred), taking(3))
    data = range(20)
    expected = list(take(3, tz_filter(pred, tz_map(func, data))))
    assert transduce(xf, collect, [], data) == expected


def test_partitioning_natural_end_flushes_partial_tail():
    xf = partitioning(2)
    for data in ([], [1], [1, 2], [1, 2, 3], [1, 2, 3, 4, 5]):
        assert transduce(xf, collect, [], data) == list(partition_all(2, data))


def test_take_does_not_over_consume_upstream():
    pulled = []

    def source():
        for i in range(10):
            pulled.append(i)
            yield i

    result = transduce(taking(3), collect, [], source())
    assert result == [0, 1, 2]
    assert pulled == [0, 1, 2]


def test_take_zero_pulls_nothing():
    pulled = []

    def source():
        for i in range(10):
            pulled.append(i)
            yield i

    assert transduce(taking(0), collect, [], source()) == []
    assert pulled == []


def test_partition_short_circuit_keeps_delivered_boundary():
    # take(3) stops the run with one item still buffered in partition;
    # the partial tail is discarded rather than emitted downstream.
    xf = compose(taking(3), partitioning(2))
    assert transduce(xf, collect, [], [1, 2, 3, 4, 5]) == [(1, 2)]
    # The iterator composition sees the tail because partition_all
    # drains its (already truncated) input fully.
    assert list(partition_all(2, take(3, [1, 2, 3, 4, 5]))) == [(1, 2), (3,)]


def test_partition_short_circuit_never_emits_beyond_take():
    xf = compose(taking(1), partitioning(3))
    assert transduce(xf, collect, [], [1, 2, 3, 4, 5]) == []


def test_empty_input_call_sequence():
    run = Logged([])
    result = transduce(mapping(str), run.step, [], run.input(), complete=run.complete)
    assert result == []
    assert run.log == [('complete', ())]


def test_last_partial_batch_call_sequence():
    run = Logged([1, 2, 3])
    xf = partitioning(2)
    result = transduce(xf, run.step, [], run.input(), complete=run.complete)
    assert result == [(1, 2), (3,)]
    assert run.log == [
        ('pull', 1),
        ('pull', 2),
        ('step', (1, 2)),
        ('pull', 3),
        ('step', (3,)),
        ('complete', ((1, 2), (3,))),
    ]
    assert result == list(partition_all(2, [1, 2, 3]))


def test_reducer_initiated_stop_call_sequence():
    run = Logged([1, 2, 3, 4, 5], stop_at=2)
    result = transduce(mapping(str), run.step, [], run.input(), complete=run.complete)
    assert result == ['1', '2']
    # Nothing is pulled upstream after the reducer asks to stop.
    assert run.log == [
        ('pull', 1),
        ('step', '1'),
        ('pull', 2),
        ('step', '2'),
        ('reduce', '2'),
        ('complete', ('1', '2')),
    ]


def test_complete_called_exactly_once_on_natural_end():
    calls = []
    xf = compose(mapping(str), taking(5))
    result = transduce(xf, collect, [], [1, 2], complete=lambda s: calls.append(s) or s)
    assert result == ['1', '2']
    assert calls == [['1', '2']]


def test_complete_called_exactly_once_on_short_circuit():
    calls = []
    xf = compose(mapping(str), taking(1))
    result = transduce(xf, collect, [], [1, 2, 3], complete=lambda s: calls.append(s) or s)
    assert result == ['1']
    assert calls == [['1']]


def test_complete_called_exactly_once_on_reducer_stop():
    calls = []

    def step(state, item):
        return reduced(state) if item == 2 else state + [item]

    result = transduce(filtering(bool), step, [], [1, 2, 3],
                       complete=lambda s: calls.append(s) or s)
    assert result == [1]
    assert calls == [[1]]


def test_complete_return_value_is_result():
    result = transduce(mapping(str), collect, [], [1, 2], complete=tuple)
    assert result == ('1', '2')


def test_exceptions_from_step_propagate():
    def step(state, item):
        raise ValueError('boom')

    with pytest.raises(ValueError, match='boom'):
        transduce(mapping(str), step, [], [1, 2, 3])


def test_exceptions_from_complete_propagate():
    def complete(state):
        raise ValueError('boom')

    with pytest.raises(ValueError, match='boom'):
        transduce(mapping(str), collect, [], [1, 2], complete=complete)


def test_exceptions_from_transform_propagate():
    def func(item):
        raise ValueError('boom')

    with pytest.raises(ValueError, match='boom'):
        transduce(mapping(func), collect, [], [1])


def test_composed_xform_is_reusable_without_state_leakage():
    xf = compose(partitioning(2), taking(2))
    first = xf.transduce(collect, [], [1, 2, 3, 4, 5])
    second = xf.transduce(collect, [], ['a', 'b', 'c'])
    third = xf.transduce(collect, [], [])
    assert first == [(1, 2), (3, 4)]
    # take counts batches here, so the flushed tail is the second batch,
    # matching take(2, partition_all(2, ...)).
    assert second == [('a', 'b'), ('c',)]
    assert second == list(take(2, partition_all(2, ['a', 'b', 'c'])))
    assert third == []
    # A short-circuited run must not leave a dangling partial batch.
    xf2 = compose(taking(3), partitioning(2))
    assert xf2.transduce(collect, [], [1, 2, 3, 4, 5]) == [(1, 2)]
    assert xf2.transduce(collect, [], [7, 8, 9]) == [(7, 8)]


def test_xform_then_and_call():
    xf = mapping(lambda x: x * 2).then(filtering(lambda x: x > 2))
    assert isinstance(xf, XForm)
    assert xf.transduce(collect, [], [1, 2, 3]) == [4, 6]


def test_compose_with_no_xforms_is_identity():
    assert compose().transduce(collect, [], [1, 2]) == [1, 2]


def test_invalid_n():
    with pytest.raises(ValueError):
        partitioning(0)
    with pytest.raises(ValueError):
        taking(-1)
