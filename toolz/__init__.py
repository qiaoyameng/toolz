from .itertoolz import *

from .functoolz import *

from .dicttoolz import *

from .recipes import *

from functools import partial, reduce

from .transducers import Reduced, is_reduced, reduced, transduce

sorted = sorted

map = map

filter = filter

# Aliases
comp = compose

from . import curried, sandbox

functoolz._sigs.create_signature_registry()


def __getattr__(name):
    if name == "__version__":
        from importlib.metadata import version

        rv = version("toolz")
        globals()[name] = rv
        return rv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
