"""Execution-only correction: observational evaluation must not reseed training."""
from contextlib import contextmanager
from functools import wraps
import random
import numpy as np
import torch


@contextmanager
def isolated_rng():
    python=random.getstate();numpy=np.random.get_state();cpu=torch.get_rng_state().clone()
    try:yield
    finally:
        random.setstate(python);np.random.set_state(numpy);torch.set_rng_state(cpu)


def preserve_rng(function):
    @wraps(function)
    def wrapped(*args,**kwargs):
        with isolated_rng():return function(*args,**kwargs)
    return wrapped
