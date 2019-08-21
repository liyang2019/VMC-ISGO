from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from operators import sun_spin1d


class SUNSpin1DTest(tf.test.TestCase):

    def test_find_states_value_encoding(self):
        hamiltonian = sun_spin1d.SUNSpin1D(1.0, True)
        states, coeffs = hamiltonian.find_states(
            np.array([0, 1, 1, 0])
        )
        self.assertEqual(states.ndim, 2)
        self.assertEqual(coeffs.ndim, 1)
        self.assertEqual(states.shape[1], 4)
        self.assertEqual(states.shape[0], coeffs.shape[0])
        np.testing.assert_equal(
            states, np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 1, 0]]))
        np.testing.assert_equal(coeffs, np.array([-1.0, -1.0, 2.0]))

        hamiltonian = sun_spin1d.SUNSpin1D(1.0, False)
        states, coeffs = hamiltonian.find_states(
            np.array([0, 1, 1, 0])
        )
        self.assertEqual(states.ndim, 2)
        self.assertEqual(coeffs.ndim, 1)
        self.assertEqual(states.shape[1], 4)
        self.assertEqual(states.shape[0], coeffs.shape[0])
        np.testing.assert_equal(
            states, np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 1, 0]]))
        np.testing.assert_equal(coeffs, np.array([-1.0, -1.0, 1.0]))

        hamiltonian = sun_spin1d.SUNSpin1D(1.0, True)
        states, coeffs = hamiltonian.find_states(
            np.array([0, 1, 2])
        )
        self.assertEqual(states.ndim, 2)
        self.assertEqual(coeffs.ndim, 1)
        self.assertEqual(states.shape[1], 3)
        self.assertEqual(states.shape[0], coeffs.shape[0])
        np.testing.assert_equal(
            states, np.array([[1, 0, 2], [0, 2, 1], [2, 1, 0], [0, 1, 2]]))
        np.testing.assert_equal(coeffs, np.array([-1.0, -1.0, -1.0, 0.0]))

        hamiltonian = sun_spin1d.SUNSpin1D(1.0, False)
        states, coeffs = hamiltonian.find_states(
            np.array([0, 1, 2])
        )
        self.assertEqual(states.ndim, 2)
        self.assertEqual(coeffs.ndim, 1)
        self.assertEqual(states.shape[1], 3)
        self.assertEqual(states.shape[0], coeffs.shape[0])
        np.testing.assert_equal(
            states, np.array([[1, 0, 2], [0, 2, 1], [0, 1, 2]]))
        np.testing.assert_equal(coeffs, np.array([-1.0, -1.0, 0.0]))

    def test_find_states_onehot_encoding(self):
        hamiltonian = sun_spin1d.SUNSpin1D(1.0, True)
        states, coeffs = hamiltonian.find_states(
            np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
        )
        self.assertEqual(states.ndim, 3)
        self.assertEqual(coeffs.ndim, 1)
        self.assertEqual(states.shape[1], 4)
        self.assertEqual(states.shape[2], 2)
        self.assertEqual(states.shape[0], coeffs.shape[0])
        np.testing.assert_equal(
            states, np.array([[[0, 1], [1, 0], [0, 1], [1, 0]],
                              [[1, 0], [0, 1], [1, 0], [0, 1]],
                              [[1, 0], [0, 1], [0, 1], [1, 0]]]))
        np.testing.assert_equal(coeffs, np.array([-1.0, -1.0, 2.0]))

        hamiltonian = sun_spin1d.SUNSpin1D(1.0, False)
        states, coeffs = hamiltonian.find_states(
            np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
        )
        self.assertEqual(states.ndim, 3)
        self.assertEqual(coeffs.ndim, 1)
        self.assertEqual(states.shape[1], 4)
        self.assertEqual(states.shape[2], 2)
        self.assertEqual(states.shape[0], coeffs.shape[0])
        np.testing.assert_equal(
            states, np.array([[[0, 1], [1, 0], [0, 1], [1, 0]],
                              [[1, 0], [0, 1], [1, 0], [0, 1]],
                              [[1, 0], [0, 1], [0, 1], [1, 0]]]))
        np.testing.assert_equal(coeffs, np.array([-1.0, -1.0, 1.0]))

        hamiltonian = sun_spin1d.SUNSpin1D(1.0, True)
        states, coeffs = hamiltonian.find_states(
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        )
        self.assertEqual(states.ndim, 3)
        self.assertEqual(coeffs.ndim, 1)
        self.assertEqual(states.shape[1], 3)
        self.assertEqual(states.shape[2], 3)
        self.assertEqual(states.shape[0], coeffs.shape[0])
        np.testing.assert_equal(
            states, np.array([[[0, 1, 0], [1, 0, 0], [0, 0, 1]],
                              [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
                              [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
                              [[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))
        np.testing.assert_equal(coeffs, np.array([-1.0, -1.0, -1.0, 0.0]))

        hamiltonian = sun_spin1d.SUNSpin1D(1.0, False)
        states, coeffs = hamiltonian.find_states(
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        )
        self.assertEqual(states.ndim, 3)
        self.assertEqual(coeffs.ndim, 1)
        self.assertEqual(states.shape[1], 3)
        self.assertEqual(states.shape[2], 3)
        self.assertEqual(states.shape[0], coeffs.shape[0])
        np.testing.assert_equal(
            states, np.array([[[0, 1, 0], [1, 0, 0], [0, 0, 1]],
                              [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
                              [[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))
        np.testing.assert_equal(coeffs, np.array([-1.0, -1.0, 0.0]))


if __name__ == "__main__":
    tf.test.main()
