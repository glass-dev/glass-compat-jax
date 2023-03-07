'''Compatibility layer for JAX.'''

import sys


def array_namespace(*objs):
    if 'jax' not in sys.modules:
        raise TypeError
    import jax
    for obj in objs:
        if not isinstance(obj, jax.Array):
            raise TypeError
    from . import array
    return array


def random_namespace(*objs):
    if 'jax' not in sys.modules:
        raise TypeError
    import jax
    for obj in objs:
        if not isinstance(obj, jax.Array):
            raise TypeError
    from . import random
    return random


def healpix_namespace(*objs):
    if 'jax' not in sys.modules:
        raise TypeError
    import jax
    for obj in objs:
        if not isinstance(obj, jax.Array):
            raise TypeError
    from . import healpix
    return healpix
