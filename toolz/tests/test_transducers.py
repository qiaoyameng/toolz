import builtins

import pytest

from toolz import compose
from toolz.itertoolz import partition_all, take as take_iter
from toolz.transducers import (
    Reduced,
    filter,
    is_reduced,
    map,
    partition,
    reduced,
    take,
    transduce,
)


def is_even(value):
    return value % 2 == 0


def logged_source(values, events):
    events.append(("source", "start"))
    for value in values:
        events.append(("pull", value))
        yield value
    events.append(("source", "end"))


def reducer_step(events, stop_after=None):
    def step(state, item):
        events.append(("deliver", item))
        state = state + [item]
        if stop_after is not None and len(state) == stop_after:
            return Reduced(state)
        return state
    return step


def completion(events):
    def complete(state):
        events.append(("complete", tuple(state)))
        return state
    return complete


def iterator_reference(values, stop_after=None):
    events = []
    pipeline = take_iter(
        stop_after if stop_after is not None else 3,
        partition_all(
            2,
            builtins.filter(
                is_even,
                builtins.map(lambda value: value + 1,
                             logged_source(values, events)),
            ),
        ),
    )
    result = list(pipeline)
    pulls = [event for event in events if event[0] == "pull"]
    return result, pulls


def transducer_reference(values, stop_after=None):
    events = []
    transformation = compose(
        map(lambda value: value + 1),
        filter(is_even),
        partition(2),
        take(3),
    )
    result = transduce(
        transformation,
        reducer_step(events, stop_after),
        [],
        logged_source(values, events),
        completion(events),
    )
    pulls = [event for event in events if event[0] == "pull"]
    return result, pulls, events


@pytest.mark.parametrize(
    ("values", "stop_after", "expected", "expected_pulls", "expected_events"),
    [
        (
            [],
            None,
            [],
            [],
            [
                ("source", "start"),
                ("source", "end"),
                ("complete", ()),
            ],
        ),
        (
            range(6),
            None,
            [(2, 4), (6,)],
            list(range(6)),
            [
                ("source", "start"),
                ("pull", 0),
                ("pull", 1),
                ("pull", 2),
                ("pull", 3),
                ("deliver", (2, 4)),
                ("pull", 4),
                ("pull", 5),
                ("source", "end"),
                ("deliver", (6,)),
                ("complete", ((2, 4), (6,))),
            ],
        ),
        (
            range(8),
            1,
            [(2, 4)],
            list(range(4)),
            [
                ("source", "start"),
                ("pull", 0),
                ("pull", 1),
                ("pull", 2),
                ("pull", 3),
                ("deliver", (2, 4)),
                ("complete", ((2, 4),)),
            ],
        ),
    ],
)
def test_transduce_matches_iterator_combination(
    values, stop_after, expected, expected_pulls, expected_events
):
    iterator_result, iterator_pulls = iterator_reference(values, stop_after)
    result, pulls, events = transducer_reference(values, stop_after)

    assert iterator_result == expected
    assert result == expected
    assert [event[1] for event in pulls] == expected_pulls
    assert events == expected_events
    assert sum(event[0] == "complete" for event in events) == 1
    if stop_after is not None:
        assert [event[1] for event in iterator_pulls] == list(values)
        assert len(pulls) < len(iterator_pulls)


def test_reduced_is_idempotent():
    value = reduced(1)

    assert reduced(value) is value
    assert is_reduced(value)
    assert value.value == 1


def test_take_does_not_pull_after_limit():
    pulled = []

    def source():
        for value in range(10):
            pulled.append(value)
            yield value

    result = transduce(
        compose(filter(lambda value: value % 2 == 1), take(2)),
        lambda state, value: state + [value],
        [],
        source(),
    )

    assert result == [1, 3]
    assert pulled == [0, 1, 2, 3]


def test_take_zero_does_not_start_source():
    events = []

    def source():
        events.append("started")
        yield 1

    result = transduce(
        take(0),
        lambda state, value: state + [value],
        [],
        source(),
        completion(events),
    )

    assert result == []
    assert events == [("complete", ())]


def test_partition_flushes_tail_only_on_natural_completion():
    result = transduce(
        compose(partition(2), take(2)),
        lambda state, value: state + [value],
        [],
        range(5),
    )

    assert result == [(0, 1), (2, 3)]


def test_take_before_partition_does_not_flush_buffered_tail():
    pulled = []
    events = []

    def source():
        for value in range(9):
            pulled.append(value)
            yield value

    result = transduce(
        compose(take(3), partition(2)),
        lambda state, value: state + [value],
        [],
        source(),
        completion(events),
    )

    assert result == [(0, 1)]
    assert pulled == [0, 1, 2]
    assert events == [("complete", ((0, 1),))]


def test_transformation_can_be_reused_without_shared_state():
    transformation = compose(map(lambda value: value + 1), partition(2))
    step = lambda state, value: state + [value]

    first = transduce(transformation, step, [], range(3))
    second = transduce(transformation, step, [], range(3))

    assert first == [(1, 2), (3,)]
    assert second == [(1, 2), (3,)]


def test_take_transformation_can_be_reused():
    transformation = take(1)
    step = lambda state, value: state + [value]

    assert transduce(transformation, step, [], range(3)) == [0]
    assert transduce(transformation, step, [], range(3)) == [0]


def test_step_exception_is_propagated_without_completion():
    events = []

    def step(state, value):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        transduce(
            partition(2),
            step,
            [],
            range(2),
            completion(events),
        )

    assert events == []


def test_transformation_exception_is_propagated_without_completion():
    events = []

    def step(state, value):
        return state + [value]

    def fail(value):
        raise ValueError("bad value")

    with pytest.raises(ValueError, match="bad value"):
        transduce(map(fail), step, [], [1], completion(events))

    assert events == []


def test_invalid_sizes():
    with pytest.raises(ValueError, match="non-negative"):
        take(-1)
    with pytest.raises(ValueError, match="positive"):
        partition(0)
