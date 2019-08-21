"""Operators for multi-component spin system loop correlation functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple

import numpy as np

from operators.operator import Operator


class SpinLoopCorrelator(Operator):

    def __init__(self, i, j, a=None, b=None, add_sign=False):
        """Initializes an Operator for multi-component spin loop correlator.

        The spin correlator is of the form S^{a, b}_i(i...j), where i, j are
        site indices, and a, b are spin indices. S^{a, b} is a SU(N) generator
        such that S^{a, b}|c> = \delta_{bc}|a>. (i...j) is a loop permutation
        operator, such that
            (i...j)|...a_ia_{i + 1}...a_j...> = |...a_ja_i...a_{j - 1}...>

        Args:
            i, j: The site indices.
            a, b: The spin indices.
            add_sign: If True, add back the 'Fermi Sign'.
        """
        self.i = i
        self.j = j
        self.a = a
        self.b = b
        self.add_sign = add_sign

    def _get_sign(self, state: np.ndarray) -> int:
        s = state[self.j]
        if s.ndim == 0:
            # rescale encoding
            if self.i <= self.j:
                n_diff = np.sum(state[self.i:self.j] != s)
            else:
                n_diff = (np.sum(state[:self.j] != s) +
                          np.sum(state[self.i:] != s))
        else:
            # onehot encoding
            if self.i <= self.j:
                n_diff = np.sum(np.any(state[self.i:self.j] != s, axis=-1))
            else:
                n_diff = (np.sum(np.any(state[:self.j] != s, axis=-1)) +
                          np.sum(np.any(state[self.i:] != s, axis=-1)))

        return 1 if np.mod(n_diff, 2) == 0 else -1

    def _loop_permute(self, state: np.ndarray) -> None:
        s = state[self.j].copy()
        if self.i <= self.j:
            state[self.i + 1:self.j + 1] = state[self.i:self.j]
        else:
            state[1:self.j + 1] = state[:self.j]
            state[0] = state[-1]
            state[self.i + 1:] = state[self.i:-1]
        state[self.i] = s

    def find_states(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        temp = state.copy()

        if self.b is not None and np.any(temp[self.j] != self.b):
            return temp[None, ...], np.array([0.0])

        self._loop_permute(temp)
        if self.a is not None and self.b is not None:
            temp[self.i] = self.a
        return temp[None, ...], np.array([self._get_sign(state)
                                          if self.add_sign else 1])
