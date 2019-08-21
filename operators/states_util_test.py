from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from operators import states_util


class StatesUtilTest(tf.test.TestCase):

    def test_random_state1d(self):
        state = states_util.random_state1d(5, 1)
        np.testing.assert_equal(state, np.zeros(5))

        state = states_util.random_state1d(5, 2, [2, 3])
        self.assertEqual(state.shape, (5,))
        self.assertEqual(sum(state == 0), 2)
        self.assertEqual(sum(state == 1), 3)

        state = states_util.random_state1d(5, 2, [2, 3], True)
        self.assertEqual(state.shape, (5, 2))
        self.assertEqual(sum(state[:, 0]), 2)
        self.assertEqual(sum(state[:, 1]), 3)
        np.testing.assert_equal(state.sum(axis=-1), np.ones(5))

        state = states_util.random_state1d(8, 3, [3, 5, 0])
        self.assertEqual(state.shape, (8,))
        self.assertEqual(sum(state == 0), 3)
        self.assertEqual(sum(state == 1), 5)
        self.assertEqual(sum(state == 2), 0)

        state = states_util.random_state1d(8, 3, [3, 5, 0], True)
        self.assertEqual(state.shape, (8, 3))
        self.assertEqual(sum(state[:, 0]), 3)
        self.assertEqual(sum(state[:, 1]), 5)
        self.assertEqual(sum(state[:, 2]), 0)
        np.testing.assert_equal(state.sum(axis=-1), np.ones(8))

    def test_random_state2d(self):
        state = states_util.random_state2d((3, 3), 1)
        np.testing.assert_equal(state, np.zeros((3, 3)))

        state = states_util.random_state2d((3, 3), 2, [2, 7])
        self.assertEqual(state.shape, (3, 3))
        self.assertEqual((state == 0).sum(), 2)
        self.assertEqual((state == 1).sum(), 7)

        state = states_util.random_state2d((3, 5), 2, [5, 10], True)
        self.assertEqual(state.shape, (3, 5, 2))
        self.assertEqual(state[:, :, 0].sum(), 5)
        self.assertEqual(state[:, :, 1].sum(), 10)
        np.testing.assert_equal(state.sum(axis=-1), np.ones((3, 5)))

        state = states_util.random_state2d((5, 5), 3, [10, 15, 0])
        self.assertEqual(state.shape, (5, 5))
        self.assertEqual((state == 0).sum(), 10)
        self.assertEqual((state == 1).sum(), 15)
        self.assertEqual((state == 2).sum(), 0)

        state = states_util.random_state2d((5, 5), 3, [10, 15, 0], True)
        self.assertEqual(state.shape, (5, 5, 3))
        self.assertEqual(state[:, :, 0].sum(), 10)
        self.assertEqual(state[:, :, 1].sum(), 15)
        self.assertEqual(state[:, :, 2].sum(), 0)
        np.testing.assert_equal(state.sum(axis=-1), np.ones((5, 5)))


if __name__ == "__main__":
    tf.test.main()
