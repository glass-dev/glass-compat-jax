'''Compatibility layer for JAX.'''

import sys


def array_namespace(*objs):
    if 'jax' not in sys.modules:
        raise TypeError
    import jax
    for obj in objs:
        if not isinstance(obj, jax.Array):
            raise TypeError
    from . import jax
    return jax
