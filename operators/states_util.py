"""Utils for generating states."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Sequence, Tuple

import numpy as np


def _one_hot_encoding1d(state: np.ndarray, n_spin: int) -> np.ndarray:
    """Transforms a 1D state to one-hot encoding form."""
    if state.ndim != 1:
        raise ValueError(
            "The input state {} is not one-dimensional.".format(state))
    n_sites = state.shape[0]
    state_one_hot = np.zeros((n_sites, n_spin), dtype=int)
    state_one_hot[range(n_sites), state] = 1
    return state_one_hot


def random_state1d(n_sites: int, n_spin: int, spin_nums: Sequence[int] = None,
                   one_hot: bool = False) -> np.ndarray:
    """Generates 1D random spin state.

    If spin_nums is None, the random state is generated by randomly
    select the spin value on every site. If spin_nums is not None, the
    random state will have specified number of spins on each spin component.
    The length of spin_nums should equal to the number of spin components.
    The sum of spin_nums should equal to the number of spin sites.

    Args:
        n_sites: Number of spin sites.
        n_spin: Number of spin component.
        spin_nums: The number of spins on each spin component.
        one_hot: Whether or not using state one-hot encoding.

    Returns:
        A numpy array of random state with first dimension size equals to
        n_sites.
    """

    if spin_nums is None:
        state = np.random.randint(0, n_spin, size=n_sites)
    else:
        if len(spin_nums) != n_spin:
            raise ValueError(
                "The length of spin_nums {} does not equal to the number of"
                " spin components {}.".format(len(spin_nums), n_spin))
        if sum(spin_nums) != n_sites:
            raise ValueError(
                "The sum of spin_nums {} does not equal to the number of"
                " spin sites {}.".format(sum(spin_nums), n_sites))
        state = []
        for s, spin_num in enumerate(spin_nums):
            state.append(np.ones(spin_num, dtype=int) * s)
        state = np.hstack(state)
        np.random.shuffle(state)
    if one_hot:
        state = _one_hot_encoding1d(state, n_spin)

    return state


def _one_hot_encoding2d(state: np.ndarray, n_spin: int) -> np.ndarray:
    """Transforms a 2D state to one-hot encoding form."""
    if state.ndim != 2:
        raise ValueError(
            "The input state {} is not two-dimensional.".format(state))
    shape = state.shape
    state = state.flatten()
    n_sites = state.shape[0]
    state_one_hot = np.zeros((n_sites, n_spin), dtype=int)
    state_one_hot[range(n_sites), state] = 1
    return np.reshape(state_one_hot, [shape[0], shape[1], n_spin])


def random_state2d(n_sites: Tuple[int, int], n_spin: int,
                   spin_nums: Sequence[int] = None,
                   one_hot: bool = False) -> np.ndarray:
    """Generates 2D random spin state.

    If spin_nums is None, the random state is generated by randomly
    select the spin value on every site. If spin_nums is not None, the
    random state will have specified number of spins on each spin component.
    The length of spin_nums should equal to the number of spin components.
    The sum of spin_nums should equal to the number of spin sites.

    Args:
        n_sites: Number of spin sites along x and y directions.
        n_spin: Number of spin component.
        spin_nums: The number of spins on each spin component.
        one_hot: Whether or not using state one-hot encoding.

    Returns:
        A numpy array of random state with first two dimensions' shape equals to
        n_sites.
    """
    if spin_nums is None:
        state = np.random.randint(0, n_spin, size=n_sites)
    else:
        if len(spin_nums) != n_spin:
            raise ValueError(
                "The length of spin_nums {} does not equal to the number of"
                " spin components {}.".format(
                    len(spin_nums), n_spin))
        if sum(spin_nums) != np.prod(n_sites):
            raise ValueError(
                "The sum of spin_nums {} does not equal to the number of"
                " spin sites {}.".format(sum(spin_nums), np.prod(n_sites)))
        state = []
        for s, spin_num in enumerate(spin_nums):
            state.append(np.ones(spin_num, dtype=int) * s)
        state = np.hstack(state)
        np.random.shuffle(state)
        state = np.reshape(state, n_sites)

    if one_hot:
        state = _one_hot_encoding2d(state, n_spin)

    return state
