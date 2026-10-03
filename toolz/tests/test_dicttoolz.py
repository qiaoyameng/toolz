from collections import defaultdict as _defaultdict
from collections.abc import Mapping
import os
from toolz.dicttoolz import (merge, merge_with, valmap, keymap, update_in,
                             update_in_multi, assoc, dissoc, keyfilter,
                             valfilter, itemmap, itemfilter, assoc_in, get_in)
from toolz.functoolz import identity
from toolz.utils import raises


def inc(x):
    return x + 1


def iseven(i):
    return i % 2 == 0


class TestDict:
    """Test typical usage: dict inputs, no factory keyword.

    Class attributes:
        D: callable that inputs a dict and creates or returns a MutableMapping
        kw: kwargs dict to specify "factory" keyword (if applicable)
    """
    D = dict
    kw = {}

    def test_merge(self):
        D, kw = self.D, self.kw
        assert merge(D({1: 1, 2: 2}), D({3: 4}), **kw) == D({1: 1, 2: 2, 3: 4})

    def test_merge_iterable_arg(self):
        D, kw = self.D, self.kw
        assert merge([D({1: 1, 2: 2}), D({3: 4})], **kw) == D({1: 1, 2: 2, 3: 4})

    def test_merge_with(self):
        D, kw = self.D, self.kw
        dicts = D({1: 1, 2: 2}), D({1: 10, 2: 20})
        assert merge_with(sum, *dicts, **kw) == D({1: 11, 2: 22})
        assert merge_with(tuple, *dicts, **kw) == D({1: (1, 10), 2: (2, 20)})

        dicts = D({1: 1, 2: 2, 3: 3}), D({1: 10, 2: 20})
        assert merge_with(sum, *dicts, **kw) == D({1: 11, 2: 22, 3: 3})
        assert merge_with(tuple, *dicts, **kw) == D({1: (1, 10), 2: (2, 20), 3: (3,)})

        assert not merge_with(sum)

    def test_merge_with_iterable_arg(self):
        D, kw = self.D, self.kw
        dicts = D({1: 1, 2: 2}), D({1: 10, 2: 20})
        assert merge_with(sum, *dicts, **kw) == D({1: 11, 2: 22})
        assert merge_with(sum, dicts, **kw) == D({1: 11, 2: 22})
        assert merge_with(sum, iter(dicts), **kw) == D({1: 11, 2: 22})

    def test_valmap(self):
        D, kw = self.D, self.kw
        assert valmap(inc, D({1: 1, 2: 2}), **kw) == D({1: 2, 2: 3})

    def test_keymap(self):
        D, kw = self.D, self.kw
        assert keymap(inc, D({1: 1, 2: 2}), **kw) == D({2: 1, 3: 2})

    def test_itemmap(self):
        D, kw = self.D, self.kw
        assert itemmap(reversed, D({1: 2, 2: 4}), **kw) == D({2: 1, 4: 2})

    def test_valfilter(self):
        D, kw = self.D, self.kw
        assert valfilter(iseven, D({1: 2, 2: 3}), **kw) == D({1: 2})

    def test_keyfilter(self):
        D, kw = self.D, self.kw
        assert keyfilter(iseven, D({1: 2, 2: 3}), **kw) == D({2: 3})

    def test_itemfilter(self):
        D, kw = self.D, self.kw
        assert itemfilter(lambda item: iseven(item[0]), D({1: 2, 2: 3}), **kw) == D({2: 3})
        assert itemfilter(lambda item: iseven(item[1]), D({1: 2, 2: 3}), **kw) == D({1: 2})

    def test_assoc(self):
        D, kw = self.D, self.kw
        assert assoc(D({}), "a", 1, **kw) == D({"a": 1})
        assert assoc(D({"a": 1}), "a", 3, **kw) == D({"a": 3})
        assert assoc(D({"a": 1}), "b", 3, **kw) == D({"a": 1, "b": 3})

        # Verify immutability:
        d = D({'x': 1})
        oldd = d
        assoc(d, 'x', 2, **kw)
        assert d is oldd

    def test_dissoc(self):
        D, kw = self.D, self.kw
        assert dissoc(D({"a": 1}), "a", **kw) == D({})
        assert dissoc(D({"a": 1, "b": 2}), "a", **kw) == D({"b": 2})
        assert dissoc(D({"a": 1, "b": 2}), "b", **kw) == D({"a": 1})
        assert dissoc(D({"a": 1, "b": 2}), "a", "b", **kw) == D({})
        assert dissoc(D({"a": 1}), "a", **kw) == dissoc(dissoc(D({"a": 1}), "a", **kw), "a", **kw)

        # Verify immutability:
        d = D({'x': 1})
        oldd = d
        d2 = dissoc(d, 'x', **kw)
        assert d is oldd
        assert d2 is not oldd

    def test_assoc_in(self):
        D, kw = self.D, self.kw
        assert assoc_in(D({"a": 1}), ["a"], 2, **kw) == D({"a": 2})
        assert (assoc_in(D({"a": D({"b": 1})}), ["a", "b"], 2, **kw) ==
                D({"a": D({"b": 2})}))
        assert assoc_in(D({}), ["a", "b"], 1, **kw) == D({"a": D({"b": 1})})

        # Verify immutability:
        d = D({'x': 1})
        oldd = d
        d2 = assoc_in(d, ['x'], 2, **kw)
        assert d is oldd
        assert d2 is not oldd

    def test_update_in(self):
        D, kw = self.D, self.kw
        assert update_in(D({"a": 0}), ["a"], inc, **kw) == D({"a": 1})
        assert update_in(D({"a": 0, "b": 1}), ["b"], str, **kw) == D({"a": 0, "b": "1"})
        assert (update_in(D({"t": 1, "v": D({"a": 0})}), ["v", "a"], inc, **kw) ==
                D({"t": 1, "v": D({"a": 1})}))
        # Handle one missing key.
        assert update_in(D({}), ["z"], str, None, **kw) == D({"z": "None"})
        assert update_in(D({}), ["z"], inc, 0, **kw) == D({"z": 1})
        assert update_in(D({}), ["z"], lambda x: x+"ar", default="b", **kw) == D({"z": "bar"})
        # Same semantics as Clojure for multiple missing keys, ie. recursively
        # create nested empty dictionaries to the depth specified by the
        # keys with the innermost value set to f(default).
        assert update_in(D({}), [0, 1], inc, default=-1, **kw) == D({0: D({1: 0})})
        assert update_in(D({}), [0, 1], str, default=100, **kw) == D({0: D({1: "100"})})
        assert (update_in(D({"foo": "bar", 1: 50}), ["d", 1, 0], str, 20, **kw) ==
                D({"foo": "bar", 1: 50, "d": D({1: D({0: "20"})})}))
        # Verify immutability:
        d = D({'x': 1})
        oldd = d
        update_in(d, ['x'], inc, **kw)
        assert d is oldd

    def test_update_in_multi(self):
        D, kw = self.D, self.kw
        assert update_in_multi(
            D({"a": 0, "b": 2}),
            [(["a"], inc), (["b"], inc)],
            **kw,
        ) == D({"a": 1, "b": 3})

        assert update_in_multi(
            D({"v": D({"a": 0, "b": 1})}),
            [(["v", "a"], inc), (["v", "b"], str)],
            **kw,
        ) == D({"v": D({"a": 1, "b": "1"})})

        assert update_in_multi(
            D({}),
            [(["a", "b"], inc), (["a", "c"], inc)],
            default=0,
            **kw,
        ) == D({"a": D({"b": 1, "c": 1})})

        assert update_in_multi(
            D({}),
            [(["a"], inc, 0), (["b"], str, None)],
            **kw,
        ) == D({"a": 1, "b": "None"})

    def test_factory(self):
        D, kw = self.D, self.kw
        assert merge(defaultdict(int, D({1: 2})), D({2: 3})) == {1: 2, 2: 3}
        assert (merge(defaultdict(int, D({1: 2})), D({2: 3}),
                      factory=lambda: defaultdict(int)) ==
                defaultdict(int, D({1: 2, 2: 3})))
        assert not (merge(defaultdict(int, D({1: 2})), D({2: 3}),
                          factory=lambda: defaultdict(int)) == {1: 2, 2: 3})
        assert raises(TypeError, lambda: merge(D({1: 2}), D({2: 3}), factoryy=dict))


class defaultdict(_defaultdict):
    def __eq__(self, other):
        return (super().__eq__(other) and
                isinstance(other, _defaultdict) and
                self.default_factory == other.default_factory)


class TestDefaultDict(TestDict):
    """Test defaultdict as input and factory

    Class attributes:
        D: callable that inputs a dict and creates or returns a MutableMapping
        kw: kwargs dict to specify "factory" keyword (if applicable)
    """
    @staticmethod
    def D(dict_):
        return defaultdict(int, dict_)

    kw = {'factory': lambda: defaultdict(int)}


class CustomMapping:
    """Define methods of the MutableMapping protocol required by dicttoolz"""
    def __init__(self, *args, **kwargs):
        self._d = dict(*args, **kwargs)

    def __getitem__(self, key):
        return self._d[key]

    def __setitem__(self, key, val):
        self._d[key] = val

    def __delitem__(self, key):
        del self._d[key]

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __contains__(self, key):
        return key in self._d

    def __eq__(self, other):
        return isinstance(other, CustomMapping) and self._d == other._d

    def __ne__(self, other):
        return not isinstance(other, CustomMapping) or self._d != other._d

    def keys(self):
        return self._d.keys()

    def values(self):
        return self._d.values()

    def items(self):
        return self._d.items()

    def update(self, *args, **kwargs):
        self._d.update(*args, **kwargs)

    # Unused methods that are part of the MutableMapping protocol
    #def get(self, key, *args):
    #    return self._d.get(key, *args)

    #def pop(self, key, *args):
    #    return self._d.pop(key, *args)

    #def popitem(self, key):
    #    return self._d.popitem()

    #def clear(self):
    #    self._d.clear()

    #def setdefault(self, key, *args):
    #    return self._d.setdefault(self, key, *args)


class TestCustomMapping(TestDict):
    """Test CustomMapping as input and factory

    Class attributes:
        D: callable that inputs a dict and creates or returns a MutableMapping
        kw: kwargs dict to specify "factory" keyword (if applicable)
    """
    D = CustomMapping
    kw = {'factory': lambda: CustomMapping()}


def test_environ():
    # See: https://github.com/pytoolz/cytoolz/issues/127
    assert keymap(identity, os.environ) == os.environ
    assert valmap(identity, os.environ) == os.environ
    assert itemmap(identity, os.environ) == os.environ


def test_merge_with_non_dict_mappings():
    class Foo(Mapping):
        def __init__(self, d):
            self.d = d

        def __iter__(self):
            return iter(self.d)

        def __getitem__(self, key):
            return self.d[key]

        def __len__(self):
            return len(self.d)

    d = Foo({1: 1})

    assert merge(d) is d or merge(d) == {1: 1}
    assert merge_with(sum, d) == {1: 1}


def test_merge_kwarg_error_names_the_offending_keyword():
    """The TypeError must name the keyword the caller actually passed."""
    try:
        merge({1: 2}, factoryy=dict)
        raise AssertionError("expected TypeError")
    except TypeError as e:
        assert "factoryy" in str(e)


def test_get_in_returns_default_for_a_missing_path():
    assert get_in(["a", "b"], {"a": {"b": 7}}) == 7
    assert get_in(["a", "z"], {"a": {"b": 7}}) is None
    assert get_in(["a", "z"], {"a": {"b": 7}}, default=-1) == -1
    assert get_in(["a", "b", "c"], {"a": {"b": 7}}, default=-1) == -1


def test_get_in_raises_when_no_default_is_set():
    assert raises(KeyError, lambda: get_in(["a", "z"], {"a": {"b": 7}}, no_default=True))
    assert raises(TypeError, lambda: get_in(["a", "b", "c"], {"a": {"b": 7}}, no_default=True))


def test_dissoc_agrees_on_both_sides_of_its_size_heuristic():
    """dissoc picks between two strategies on len(keys) vs len(d); both must agree."""
    d = {str(i): i for i in range(10)}
    assert dissoc(d, "1", "2") == {k: v for k, v in d.items() if k not in ("1", "2")}
    assert dissoc(d, *[str(i) for i in range(9)]) == {"9": 9}
    assert dissoc(d) == d


def test_update_in_multi_copies_shared_ancestors_once():
    untouched_leaf = {"z": 3}
    shared = {"x": {"value": 1}, "y": {"value": 2}, "z": untouched_leaf}
    original = {"a": shared, "u": untouched_leaf}
    constructions = []

    def factory():
        created = {}
        constructions.append(created)
        return created

    result = update_in_multi(
        original,
        [
            (["a", "x", "value"], inc),
            (["a", "y", "value"], inc),
        ],
        factory=factory,
    )

    assert result == {
        "a": {
            "x": {"value": 2},
            "y": {"value": 3},
            "z": {"z": 3},
        },
        "u": untouched_leaf,
    }
    assert len(constructions) == 4
    assert result["a"]["z"] is untouched_leaf
    assert result["u"] is untouched_leaf
    assert result["a"] is not original["a"]
    assert result["a"]["x"] is not original["a"]["x"]
    assert result["a"]["y"] is not original["a"]["y"]


def test_update_in_multi_copies_missing_shared_ancestor_once():
    constructions = []

    def factory():
        created = {}
        constructions.append(created)
        return created

    result = update_in_multi(
        {},
        [(["a", "x"], inc), (["a", "y"], inc)],
        default=0,
        factory=factory,
    )

    assert result == {"a": {"x": 1, "y": 1}}
    assert len(constructions) == 2


def test_update_in_multi_functions_observe_original_values():
    data = {"shared": {"a": 1, "b": 2}}
    observed = []

    def mutate_then_record(value):
        data["shared"]["b"] = 100
        observed.append(value)
        return value

    result = update_in_multi(
        data,
        [
            (["shared", "a"], mutate_then_record),
            (["shared", "b"], lambda value: observed.append(value) or value),
        ],
    )

    assert result == {"shared": {"a": 1, "b": 2}}
    assert observed == [1, 2]


def test_update_in_multi_consumes_single_pass_path_iterators():
    def one_pass_path():
        yield "a"
        yield "b"

    assert update_in_multi(
        {"a": {"b": 1}},
        [(one_pass_path(), inc)],
    ) == {"a": {"b": 2}}


def test_update_in_multi_rejects_conflicts_before_calling_functions():
    called = []

    def record(label):
        def update(value):
            called.append(label)
            return value
        return update

    conflicting_updates = [
        [(["a"], record("same")), (["a"], record("same"))],
        [
            (["a"], record("ancestor")),
            (["a", "b"], record("descendant")),
        ],
        [
            (["a", "b"], record("descendant")),
            (["a"], record("ancestor")),
        ],
    ]

    for updates in conflicting_updates:
        called.clear()
        assert raises(ValueError, lambda: update_in_multi({}, updates))
        assert called == []


def test_update_in_multi_function_failure_leaves_original_unchanged():
    untouched_leaf = {"z": 3}
    original = {"a": {"x": 1, "y": 2}, "u": untouched_leaf}
    factory_inputs = []

    class PartialResult(Exception):
        pass

    def fail(value):
        raise PartialResult

    def factory():
        created = {}
        factory_inputs.append(created)
        return created

    assert raises(
        PartialResult,
        lambda: update_in_multi(
            original,
            [(["a", "x"], inc), (["a", "y"], fail)],
            factory=factory,
        ),
    )

    assert original == {"a": {"x": 1, "y": 2}, "u": untouched_leaf}
    assert original["u"] is untouched_leaf
    assert factory_inputs == []


def test_update_in_multi_invalid_existing_path_fails_before_functions():
    called = []

    assert raises(
        TypeError,
        lambda: update_in_multi(
            {"a": 1},
            [
                (["a", "b"], lambda value: called.append(value) or value),
                (["c"], lambda value: called.append(value) or value),
            ],
        ),
    )
    assert called == []


def test_update_in_multi_deep_path_does_not_recurse():
    depth = 3000
    keys = [str(i) for i in range(depth)]
    original = {}
    node = original
    for key in keys[:-1]:
        node[key] = {}
        node = node[key]
    node[keys[-1]] = 1

    result = update_in_multi(original, [(keys, inc)])

    assert get_in(keys, result) == 2
    assert get_in(keys, original) == 1
