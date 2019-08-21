"""Two dimensional Heisenberg model on square lattice."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple

import numpy as np

from operators import operator


class Heisenberg2DSquare(operator.Operator):

    def __init__(self, pbc: bool) -> None:
        """Initializes a 2D Heisenberg AFM Hamiltonian.

        H = \sum_<i,j> S^x_iS^x_j + S^y_iS^y_j + S^z_iS^z_j

        Args:
            pbc: True for periodic boundary condition.
        """
        self._pbc = pbc
        self._nearest_neighbors = ((0, 1), (1, 0))

    def find_states(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        states = []
        coeffs = []
        n_r, n_c = state.shape[:2]
        diag = 0.0
        for r in range(n_r):
            for c in range(n_c):
                for dr, dc in self._nearest_neighbors:
                    rr, cc = r + dr, c + dc
                    if rr >= n_r or cc >= n_c:
                        if self._pbc:
                            rr %= n_r
                            cc %= n_c
                        else:
                            continue
                    if np.any(state[r, c] != state[rr, cc]):
                        temp = state.copy()

                        # This is the correct way of swapping states when
                        # temp.ndim > 2.
                        temp[[r, rr], [c, cc]] = temp[[rr, r], [cc, c]]
                        states.append(temp)
                        coeffs.append(-0.5)
                        diag -= 0.25
                    else:
                        diag += 0.25
        states.append(state.copy())
        coeffs.append(diag)
        return np.stack(states), np.array(coeffs)
