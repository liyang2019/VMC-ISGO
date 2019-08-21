"""Bethe Ansatz for solving the ground state energies of SU(N) spin chains."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Callable, Sequence

import numpy as np
from scipy.optimize import root


def _get_bethe_ansatz_equations(
        spin_nums: Sequence[int]) -> Callable[[np.ndarray], np.ndarray]:
    """Gets the Bethe Ansatz equations.

    Args:
        spin_nums: The number of spins on each spin component.

    Returns:
        A callable representing the Bethe Ansatz equations. The input is a
        concatenation of all the rapidities with j in increasing order as in
        Eq. (1) in the supplemental material.
    """
    assert all([spin_nums[0] == num for num in spin_nums])
    N = sum(spin_nums)
    M = spin_nums[0]
    P = len(spin_nums)
    Js = []
    Ms = []
    for i in range(1, P):
        Mi = (P - i) * M
        Js.append(np.linspace(-(Mi - 1) / 2, (Mi - 1) / 2, Mi))
        Ms.append(Mi)
    Js = -np.pi * np.hstack(Js)

    def bethe_ansatz_equations(rapidities: np.ndarray) -> np.ndarray:
        """The Bethe Ansatz Equations."""
        res = Js.copy()
        s = 0
        for j in range(P - 1):
            e = s + Ms[j]
            curr = rapidities[s:e][:, None]
            if j == 0:
                res[s:e] += N * np.arctan(2 * curr[:, 0])
            else:
                prev = rapidities[s - Ms[j - 1]:s][:, None]
                res[s:e] += np.arctan(2 * (curr - prev.T)).sum(axis=1)
            res[s:e] -= np.arctan(curr - curr.T).sum(axis=1)
            if e < len(rapidities):
                next = rapidities[e:e + Ms[j + 1]][:, None]
                res[s:e] += np.arctan(2 * (curr - next.T)).sum(axis=1)
            s = e
        return res

    return bethe_ansatz_equations


def _solve_ground_state_rapidities(spin_nums: Sequence[int]) -> np.ndarray:
    """Solves ground state rapidities."""
    fun = _get_bethe_ansatz_equations(spin_nums)
    M = spin_nums[0]
    P = len(spin_nums)
    return root(fun, np.zeros(P * (P - 1) * M // 2)).x


def _compute_ground_state_energy(spin_nums: Sequence[int],
                                 rapidities: np.ndarray) -> np.ndarray:
    """Computes ground state energy."""
    N = sum(spin_nums)
    M = spin_nums[0]
    P = len(spin_nums)
    return N - np.sum(1 / (rapidities[:(P - 1) * M] ** 2 + 0.25))


def compute_ground_state_energy(spin_nums: Sequence[int]) -> float:
    """Compute ground state energy for SU(N) spin chain using Bethe Ansatz."""
    rapidities = _solve_ground_state_rapidities(spin_nums)
    E = _compute_ground_state_energy(spin_nums, rapidities)
    return float(E)
