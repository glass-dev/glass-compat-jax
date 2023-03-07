'''Random namespace for JAX.'''

# FIXME implement the functions below
raise NotImplementedError

import jax
import jax.random

RandomState = jax.Array


def random_state(seed=None):
    return jax.random.default_rng(seed)


def split(rand):
    return rand, rand


def standard_normal(rand, shape=()):
    return jax.random.normal(...)


def choice(rand, shape=(), replace=True, p=None, axis=0):
    return jax.random.choice(...)


def uniform(rand, low=0.0, high=1.0, shape=()):
    return jax.random.uniform(...)


def normal(rand, loc=0.0, scale=1.0, shape=()):
    rand, norm = jax.random.normal(...)
    # transform norm
    return rand, norm


def poisson(rand, lam=1.0, shape=()):
    return jax.random.poisson(...)
