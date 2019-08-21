from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from operators import spin_loop_correlator


class SpinLoopCorrelatorTest(tf.test.TestCase):

    def test_find_states_value_encoding(self):
        loop_correlator = spin_loop_correlator.SpinLoopCorrelator(1, 3)
        states, coeffs = loop_correlator.find_states(np.array([0, 1, 3, 3, 4]))
        self.assertEqual(states.shape, (1, 5))
        self.assertEqual(coeffs.shape, (1,))
        np.testing.assert_equal(states, np.array([[0, 3, 1, 3, 4]]))
        np.testing.assert_equal(coeffs, 1.0)

        loop_correlator = spin_loop_correlator.SpinLoopCorrelator(3, 1)
        states, coeffs = loop_correlator.find_states(np.array([0, 1, 3, 3, 4]))
        self.assertEqual(states.shape, (1, 5))
        self.assertEqual(coeffs.shape, (1,))
        np.testing.assert_equal(states, np.array([[4, 0, 3, 1, 3]]))
        np.testing.assert_equal(coeffs, 1.0)

        loop_correlator = spin_loop_correlator.SpinLoopCorrelator(1, 3,
                                                                  add_sign=True)
        states, coeffs = loop_correlator.find_states(np.array([0, 1, 3, 3, 4]))
        self.assertEqual(states.shape, (1, 5))
        self.assertEqual(coeffs.shape, (1,))
        np.testing.assert_equal(states, np.array([[0, 3, 1, 3, 4]]))
        np.testing.assert_equal(coeffs, -1.0)

        loop_correlator = spin_loop_correlator.SpinLoopCorrelator(3, 1,
                                                                  add_sign=True)
        states, coeffs = loop_correlator.find_states(np.array([0, 1, 3, 3, 4]))
        self.assertEqual(states.shape, (1, 5))
        self.assertEqual(coeffs.shape, (1,))
        np.testing.assert_equal(states, np.array([[4, 0, 3, 1, 3]]))
        np.testing.assert_equal(coeffs, -1.0)

    def test_find_states_onehot_encoding(self):
        loop_correlator = spin_loop_correlator.SpinLoopCorrelator(1, 3)
        states, coeffs = loop_correlator.find_states(np.array(
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0],
             [0, 0, 0, 0, 1]]))
        self.assertEqual(states.shape, (1, 5, 5))
        self.assertEqual(coeffs.shape, (1,))
        np.testing.assert_equal(states, np.array([[[1, 0, 0, 0, 0],
                                                   [0, 0, 0, 1, 0],
                                                   [0, 1, 0, 0, 0],
                                                   [0, 0, 0, 1, 0],
                                                   [0, 0, 0, 0, 1]]]))
        np.testing.assert_equal(coeffs, 1.0)

        loop_correlator = spin_loop_correlator.SpinLoopCorrelator(3, 1)
        states, coeffs = loop_correlator.find_states(np.array(
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0],
             [0, 0, 0, 0, 1]]))
        self.assertEqual(states.shape, (1, 5, 5))
        self.assertEqual(coeffs.shape, (1,))
        np.testing.assert_equal(states, np.array([[[0, 0, 0, 0, 1],
                                                   [1, 0, 0, 0, 0],
                                                   [0, 0, 0, 1, 0],
                                                   [0, 1, 0, 0, 0],
                                                   [0, 0, 0, 1, 0]]]))
        np.testing.assert_equal(coeffs, 1.0)

        loop_correlator = spin_loop_correlator.SpinLoopCorrelator(1, 3,
                                                                  add_sign=True)
        states, coeffs = loop_correlator.find_states(np.array(
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0],
             [0, 0, 0, 0, 1]]))
        self.assertEqual(states.shape, (1, 5, 5))
        self.assertEqual(coeffs.shape, (1,))
        np.testing.assert_equal(states, np.array([[[1, 0, 0, 0, 0],
                                                   [0, 0, 0, 1, 0],
                                                   [0, 1, 0, 0, 0],
                                                   [0, 0, 0, 1, 0],
                                                   [0, 0, 0, 0, 1]]]))
        np.testing.assert_equal(coeffs, -1.0)

        loop_correlator = spin_loop_correlator.SpinLoopCorrelator(3, 1,
                                                                  add_sign=True)
        states, coeffs = loop_correlator.find_states(np.array(
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0],
             [0, 0, 0, 0, 1]]))
        self.assertEqual(states.shape, (1, 5, 5))
        self.assertEqual(coeffs.shape, (1,))
        np.testing.assert_equal(states, np.array([[[0, 0, 0, 0, 1],
                                                   [1, 0, 0, 0, 0],
                                                   [0, 0, 0, 1, 0],
                                                   [0, 1, 0, 0, 0],
                                                   [0, 0, 0, 1, 0]]]))
        np.testing.assert_equal(coeffs, -1.0)


if __name__ == "__main__":
    tf.test.main()
