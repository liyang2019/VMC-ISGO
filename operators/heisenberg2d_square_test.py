from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from operators import heisenberg2d_square


class Heisenberg1DTest(tf.test.TestCase):

    def test_find_states_value_encoding(self):
        hamiltonian = heisenberg2d_square.Heisenberg2DSquare(True)
        states, coeffs = hamiltonian.find_states(
            np.array([[0, 1, 1],
                      [0, 1, 0],
                      [1, 0, 0]])
        )
        self.assertEqual(states.ndim, 3)
        self.assertEqual(coeffs.ndim, 1)
        self.assertEqual(states.shape[1], 3)
        self.assertEqual(states.shape[2], 3)
        self.assertEqual(states.shape[0], coeffs.shape[0])
        np.testing.assert_equal(
            states, np.array([[[1, 0, 1],
                               [0, 1, 0],
                               [1, 0, 0]],
                              [[1, 1, 0],
                               [0, 1, 0],
                               [1, 0, 0]],
                              [[0, 1, 0],
                               [0, 1, 1],
                               [1, 0, 0]],
                              [[0, 1, 1],
                               [1, 0, 0],
                               [1, 0, 0]],
                              [[0, 1, 1],
                               [1, 1, 0],
                               [0, 0, 0]],
                              [[0, 1, 1],
                               [0, 0, 1],
                               [1, 0, 0]],
                              [[0, 1, 1],
                               [0, 0, 0],
                               [1, 1, 0]],
                              [[0, 1, 1],
                               [0, 1, 0],
                               [0, 1, 0]],
                              [[1, 1, 1],
                               [0, 1, 0],
                               [0, 0, 0]],
                              [[0, 0, 1],
                               [0, 1, 0],
                               [1, 1, 0]],
                              [[0, 1, 1],
                               [0, 1, 0],
                               [0, 0, 1]],
                              [[0, 1, 0],
                               [0, 1, 0],
                               [1, 0, 1]],
                              [[0, 1, 1],
                               [0, 1, 0],
                               [1, 0, 0]]
                              ]))
        np.testing.assert_equal(
            coeffs, np.array(
                [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
                 -0.5, -0.5, -1.5]))

    def test_find_states_onehot_encoding(self):
        hamiltonian = heisenberg2d_square.Heisenberg2DSquare(True)
        states, coeffs = hamiltonian.find_states(
            np.array([[[1, 0], [0, 1], [0, 1]],
                      [[1, 0], [0, 1], [1, 0]],
                      [[0, 1], [1, 0], [1, 0]]])
        )
        self.assertEqual(states.ndim, 4)
        self.assertEqual(coeffs.ndim, 1)
        self.assertEqual(states.shape[1], 3)
        self.assertEqual(states.shape[2], 3)
        self.assertEqual(states.shape[3], 2)
        self.assertEqual(states.shape[0], coeffs.shape[0])
        np.testing.assert_equal(
            states, np.array([[[[0, 1], [1, 0], [0, 1]],
                               [[1, 0], [0, 1], [1, 0]],
                               [[0, 1], [1, 0], [1, 0]]],
                              [[[0, 1], [0, 1], [1, 0]],
                               [[1, 0], [0, 1], [1, 0]],
                               [[0, 1], [1, 0], [1, 0]]],
                              [[[1, 0], [0, 1], [1, 0]],
                               [[1, 0], [0, 1], [0, 1]],
                               [[0, 1], [1, 0], [1, 0]]],
                              [[[1, 0], [0, 1], [0, 1]],
                               [[0, 1], [1, 0], [1, 0]],
                               [[0, 1], [1, 0], [1, 0]]],
                              [[[1, 0], [0, 1], [0, 1]],
                               [[0, 1], [0, 1], [1, 0]],
                               [[1, 0], [1, 0], [1, 0]]],
                              [[[1, 0], [0, 1], [0, 1]],
                               [[1, 0], [1, 0], [0, 1]],
                               [[0, 1], [1, 0], [1, 0]]],
                              [[[1, 0], [0, 1], [0, 1]],
                               [[1, 0], [1, 0], [1, 0]],
                               [[0, 1], [0, 1], [1, 0]]],
                              [[[1, 0], [0, 1], [0, 1]],
                               [[1, 0], [0, 1], [1, 0]],
                               [[1, 0], [0, 1], [1, 0]]],
                              [[[0, 1], [0, 1], [0, 1]],
                               [[1, 0], [0, 1], [1, 0]],
                               [[1, 0], [1, 0], [1, 0]]],
                              [[[1, 0], [1, 0], [0, 1]],
                               [[1, 0], [0, 1], [1, 0]],
                               [[0, 1], [0, 1], [1, 0]]],
                              [[[1, 0], [0, 1], [0, 1]],
                               [[1, 0], [0, 1], [1, 0]],
                               [[1, 0], [1, 0], [0, 1]]],
                              [[[1, 0], [0, 1], [1, 0]],
                               [[1, 0], [0, 1], [1, 0]],
                               [[0, 1], [1, 0], [0, 1]]],
                              [[[1, 0], [0, 1], [0, 1]],
                               [[1, 0], [0, 1], [1, 0]],
                               [[0, 1], [1, 0], [1, 0]]]
                              ]))
        np.testing.assert_equal(
            coeffs, np.array(
                [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
                 -0.5, -0.5, -1.5]))


if __name__ == "__main__":
    tf.test.main()
