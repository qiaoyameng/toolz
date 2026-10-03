import operator
import collections
from functools import reduce
from collections.abc import Mapping

__all__ = ('merge', 'merge_with', 'valmap', 'keymap', 'itemmap',
           'valfilter', 'keyfilter', 'itemfilter',
           'assoc', 'dissoc', 'assoc_in', 'update_in', 'update_in_many',
           'get_in')


def _get_factory(f, kwargs):
    factory = kwargs.pop('factory', dict)
    if kwargs:
        raise TypeError("{}() got an unexpected keyword argument "
                        "'{}'".format(f.__name__, kwargs.popitem()[0]))
    return factory


def merge(*dicts, **kwargs):
    """ Merge a collection of dictionaries

    >>> merge({1: 'one'}, {2: 'two'})
    {1: 'one', 2: 'two'}

    Later dictionaries have precedence

    >>> merge({1: 2, 3: 4}, {3: 3, 4: 4})
    {1: 2, 3: 3, 4: 4}

    See Also:
        merge_with
    """
    if len(dicts) == 1 and not isinstance(dicts[0], Mapping):
        dicts = dicts[0]
    factory = _get_factory(merge, kwargs)

    rv = factory()
    for d in dicts:
        rv.update(d)
    return rv


def merge_with(func, *dicts, **kwargs):
    """ Merge dictionaries and apply function to combined values

    A key may occur in more than one dict, and all values mapped from the key
    will be passed to the function as a list, such as func([val1, val2, ...]).

    >>> merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20})
    {1: 11, 2: 22}

    >>> merge_with(first, {1: 1, 2: 2}, {2: 20, 3: 30})  # doctest: +SKIP
    {1: 1, 2: 2, 3: 30}

    See Also:
        merge
    """
    if len(dicts) == 1 and not isinstance(dicts[0], Mapping):
        dicts = dicts[0]
    factory = _get_factory(merge_with, kwargs)

    values = collections.defaultdict(lambda: [].append)
    for d in dicts:
        for k, v in d.items():
            values[k](v)

    result = factory()
    for k, v in values.items():
        result[k] = func(v.__self__)
    return result


def valmap(func, d, factory=dict):
    """ Apply function to values of dictionary

    >>> bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    >>> valmap(sum, bills)  # doctest: +SKIP
    {'Alice': 65, 'Bob': 45}

    See Also:
        keymap
        itemmap
    """
    rv = factory()
    rv.update(zip(d.keys(), map(func, d.values())))
    return rv


def keymap(func, d, factory=dict):
    """ Apply function to keys of dictionary

    >>> bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    >>> keymap(str.lower, bills)  # doctest: +SKIP
    {'alice': [20, 15, 30], 'bob': [10, 35]}

    See Also:
        valmap
        itemmap
    """
    rv = factory()
    rv.update(zip(map(func, d.keys()), d.values()))
    return rv


def itemmap(func, d, factory=dict):
    """ Apply function to items of dictionary

    >>> accountids = {"Alice": 10, "Bob": 20}
    >>> itemmap(reversed, accountids)  # doctest: +SKIP
    {10: "Alice", 20: "Bob"}

    See Also:
        keymap
        valmap
    """
    rv = factory()
    rv.update(map(func, d.items()))
    return rv


def valfilter(predicate, d, factory=dict):
    """ Filter items in dictionary by value

    >>> iseven = lambda x: x % 2 == 0
    >>> d = {1: 2, 2: 3, 3: 4, 4: 5}
    >>> valfilter(iseven, d)
    {1: 2, 3: 4}

    See Also:
        keyfilter
        itemfilter
        valmap
    """
    rv = factory()
    for k, v in d.items():
        if predicate(v):
            rv[k] = v
    return rv


def keyfilter(predicate, d, factory=dict):
    """ Filter items in dictionary by key

    >>> iseven = lambda x: x % 2 == 0
    >>> d = {1: 2, 2: 3, 3: 4, 4: 5}
    >>> keyfilter(iseven, d)
    {2: 3, 4: 5}

    See Also:
        valfilter
        itemfilter
        keymap
    """
    rv = factory()
    for k, v in d.items():
        if predicate(k):
            rv[k] = v
    return rv


def itemfilter(predicate, d, factory=dict):
    """ Filter items in dictionary by item

    >>> def isvalid(item):
    ...     k, v = item
    ...     return k % 2 == 0 and v < 4

    >>> d = {1: 2, 2: 3, 3: 4, 4: 5}
    >>> itemfilter(isvalid, d)
    {2: 3}

    See Also:
        keyfilter
        valfilter
        itemmap
    """
    rv = factory()
    for item in d.items():
        if predicate(item):
            k, v = item
            rv[k] = v
    return rv


def assoc(d, key, value, factory=dict):
    """ Return a new dict with new key value pair

    New dict has d[key] set to value. Does not modify the initial dictionary.

    >>> assoc({'x': 1}, 'x', 2)
    {'x': 2}
    >>> assoc({'x': 1}, 'y', 3)   # doctest: +SKIP
    {'x': 1, 'y': 3}
    """
    d2 = factory()
    d2.update(d)
    d2[key] = value
    return d2


def dissoc(d, *keys, **kwargs):
    """ Return a new dict with the given key(s) removed.

    New dict has d[key] deleted for each supplied key.
    Does not modify the initial dictionary.

    >>> dissoc({'x': 1, 'y': 2}, 'y')
    {'x': 1}
    >>> dissoc({'x': 1, 'y': 2}, 'y', 'x')
    {}
    >>> dissoc({'x': 1}, 'y') # Ignores missing keys
    {'x': 1}
    """
    factory = _get_factory(dissoc, kwargs)
    d2 = factory()

    if len(keys) < len(d) * .6:
        d2.update(d)
        for key in keys:
            if key in d2:
                del d2[key]
    else:
        remaining = set(d)
        remaining.difference_update(keys)
        for k in remaining:
            d2[k] = d[k]
    return d2


def assoc_in(d, keys, value, factory=dict):
    """ Return a new dict with new, potentially nested, key value pair

    >>> purchase = {'name': 'Alice',
    ...             'order': {'items': ['Apple', 'Orange'],
    ...                       'costs': [0.50, 1.25]},
    ...             'credit card': '5555-1234-1234-1234'}
    >>> assoc_in(purchase, ['order', 'costs'], [0.25, 1.00]) # doctest: +SKIP
    {'credit card': '5555-1234-1234-1234',
     'name': 'Alice',
     'order': {'costs': [0.25, 1.00], 'items': ['Apple', 'Orange']}}
    """
    return update_in(d, keys, lambda x: value, value, factory)


def update_in(d, keys, func, default=None, factory=dict):
    """ Update value in a (potentially) nested dictionary

    inputs:
    d - dictionary on which to operate
    keys - list or tuple giving the location of the value to be changed in d
    func - function to operate on that value

    If keys == [k0,..,kX] and d[k0]..[kX] == v, update_in returns a copy of the
    original dictionary with v replaced by func(v), but does not mutate the
    original dictionary.

    If k0 is not a key in d, update_in creates nested dictionaries to the depth
    specified by the keys, with the innermost value set to func(default).

    >>> inc = lambda x: x + 1
    >>> update_in({'a': 0}, ['a'], inc)
    {'a': 1}

    >>> transaction = {'name': 'Alice',
    ...                'purchase': {'items': ['Apple', 'Orange'],
    ...                             'costs': [0.50, 1.25]},
    ...                'credit card': '5555-1234-1234-1234'}
    >>> update_in(transaction, ['purchase', 'costs'], sum) # doctest: +SKIP
    {'credit card': '5555-1234-1234-1234',
     'name': 'Alice',
     'purchase': {'costs': 1.75, 'items': ['Apple', 'Orange']}}

    >>> # updating a value when k0 is not in d
    >>> update_in({}, [1, 2, 3], str, default="bar")
    {1: {2: {3: 'bar'}}}
    >>> update_in({1: 'foo'}, [2, 3, 4], inc, 0)
    {1: 'foo', 2: {3: {4: 1}}}
    """
    ks = iter(keys)
    k = next(ks)

    rv = inner = factory()
    rv.update(d)

    for key in ks:
        if k in d:
            d = d[k]
            dtemp = factory()
            dtemp.update(d)
        else:
            d = dtemp = factory()

        inner[k] = inner = dtemp
        k = key

    if k in d:
        inner[k] = func(d[k])
    else:
        inner[k] = func(default)
    return rv


def update_in_many(d, updates, default=None, factory=dict):
    """ Update many values in a (potentially) nested dictionary at once

    inputs:
    d - dictionary on which to operate
    updates - iterable of (keys, func) pairs, where keys is a list, tuple,
        or iterator giving the location of the value to be changed in d,
        and func is the function to operate on that value

    Like calling ``update_in`` once for each pair, but more efficient:
    dictionaries that are ancestors of more than one updated value are
    copied only once, and branches that are not touched are shared with
    the original dictionary.

    Every func receives the value found in the original dictionary d
    (or ``default`` if its path is missing), i.e. all updates observe
    the state before any of them is applied.

    Paths must not conflict: the same path twice, or one path being a
    prefix of another, raises ValueError before any func is called.

    The original dictionary is never mutated.  If a func raises, the
    exception propagates and no partially updated result is produced.

    >>> inc = lambda x: x + 1
    >>> update_in_many({'a': 0, 'b': {'c': 1, 'd': 2}},
    ...                [(['a'], inc), (['b', 'c'], str)])
    {'a': 1, 'b': {'c': '1', 'd': 2}}

    >>> update_in_many({}, [([1, 2], str), (['x'], inc)], default=0)
    {1: {2: '0'}, 'x': 1}

    See Also:
        update_in
    """
    # Build a trie of the updates.  Each node is [func, children] where
    # func is None for purely intermediate nodes.  Materializing the
    # paths here consumes one-shot iterators exactly once and detects
    # conflicts before any func is called.
    root = [None, {}]
    for keys, func in updates:
        node = root
        depth = 0
        for k in keys:
            if node[0] is not None:
                raise ValueError("update_in_many got conflicting paths: "
                                 "one path is a prefix of another")
            node = node[1].setdefault(k, [None, {}])
            depth += 1
        if depth == 0:
            raise ValueError("update_in_many got an empty path")
        if node[0] is not None:
            raise ValueError("update_in_many got conflicting paths: "
                             "duplicate path")
        if node[1]:
            raise ValueError("update_in_many got conflicting paths: "
                             "one path is a prefix of another")
        node[0] = func

    # Copy d and, top-down, every node on the way to an updated value.
    # Iterative so that deep paths don't hit the recursion limit.
    rv = factory()
    rv.update(d)
    leaves = []
    stack = [(d, rv, root)]
    empty = {}
    while stack:
        old, new, trie = stack.pop()
        for k, child in trie[1].items():
            if k in old:
                child_old = old[k]
                present = True
            else:
                child_old = empty
                present = False
            if child[1]:
                child_new = factory()
                child_new.update(child_old)
                new[k] = child_new
                stack.append((child_old, child_new, child))
            else:
                leaves.append((new, k, child[0], present, child_old))

    # Apply the update functions last: every func sees the original
    # values, and a failing func leaves d untouched with no partial
    # result escaping.
    for new, k, func, present, old_value in leaves:
        if present:
            new[k] = func(old_value)
        else:
            new[k] = func(default)
    return rv


def get_in(keys, coll, default=None, no_default=False):
    """ Returns coll[i0][i1]...[iX] where [i0, i1, ..., iX]==keys.

    If coll[i0][i1]...[iX] cannot be found, returns ``default``, unless
    ``no_default`` is specified, then it raises KeyError or IndexError.

    ``get_in`` is a generalization of ``operator.getitem`` for nested data
    structures such as dictionaries and lists.

    >>> transaction = {'name': 'Alice',
    ...                'purchase': {'items': ['Apple', 'Orange'],
    ...                             'costs': [0.50, 1.25]},
    ...                'credit card': '5555-1234-1234-1234'}
    >>> get_in(['purchase', 'items', 0], transaction)
    'Apple'
    >>> get_in(['name'], transaction)
    'Alice'
    >>> get_in(['purchase', 'total'], transaction)
    >>> get_in(['purchase', 'items', 'apple'], transaction)
    >>> get_in(['purchase', 'items', 10], transaction)
    >>> get_in(['purchase', 'total'], transaction, 0)
    0
    >>> get_in(['y'], {}, no_default=True)
    Traceback (most recent call last):
        ...
    KeyError: 'y'

    See Also:
        itertoolz.get
        operator.getitem
    """
    try:
        return reduce(operator.getitem, keys, coll)
    except (KeyError, IndexError, TypeError):
        if no_default:
            raise
        return default
