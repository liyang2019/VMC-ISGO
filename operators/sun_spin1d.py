"""SU(N) symmetric 1D spin chain model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple

import numpy as np

from operators import operator


class SUNSpin1D(operator.Operator):

    def __init__(self, t: float, pbc: bool) -> None:
        """Initializes a SU(N) symmetric 1D spin chain Hamiltonian.

        H = t\sum_i P_{i, i+1}, where P is spin exchange operator.

        Args:
            t: The pre-factor of the Hamiltonian, if t is negative, the
                Hamiltonian is Ferromagnetic, if positive it is
                Anti-Ferromagnetic.
            pbc: True for periodic boundary condition.
        """
        self._pbc = pbc
        self._t = t

    def find_states(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        states = []
        coeffs = []
        n_sites = len(state)
        diag = 0.0
        for i in range(n_sites - 1):
            if np.any(state[i] != state[i + 1]):
                temp = state.copy()

                # This is the correct way of swapping states when temp.ndim > 1.
                temp[[i, i + 1]] = temp[[i + 1, i]]
                states.append(temp)
                coeffs.append(-self._t)
            else:
                diag += self._t
        if self._pbc:
            if np.any(state[n_sites - 1] != state[0]):
                temp = state.copy()

                # This is the correct way of swapping states when temp.ndim > 1.
                temp[[n_sites - 1, 0]] = temp[[0, n_sites - 1]]
                states.append(temp)
                coeffs.append(-self._t)
            else:
                diag += self._t
        states.append(state.copy())
        coeffs.append(diag)
        return np.stack(states), np.array(coeffs)
