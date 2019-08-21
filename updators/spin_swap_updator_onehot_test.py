from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from updators import spin_swap_updator_onehot


class SpinSwapUpdatorValueTest(tf.test.TestCase):

    def test_masks_placeholder(self):
        updator = spin_swap_updator_onehot.SpinSwapUpdatorOneHot((10, 2))
        mask_placeholder = updator.masks_placeholder()
        self.assertEqual(mask_placeholder.shape.as_list(), [None, 10])

        updator = spin_swap_updator_onehot.SpinSwapUpdatorOneHot((5, 5, 3))
        mask_placeholder = updator.masks_placeholder()
        self.assertEqual(mask_placeholder.shape.as_list(), [None, 25])

    def test_generate_masks(self):
        updator = spin_swap_updator_onehot.SpinSwapUpdatorOneHot((10, 2))
        masks = updator.generate_masks(10)
        self.assertEqual(masks.shape, (10, 10))

        updator = spin_swap_updator_onehot.SpinSwapUpdatorOneHot((5, 5, 3))
        masks = updator.generate_masks(15)
        self.assertEqual(masks.shape, (15, 25))

    def test_call(self):
        with tf.Session() as sess:
            updator = spin_swap_updator_onehot.SpinSwapUpdatorOneHot((3, 2))
            state = tf.constant([[1, 0], [0, 1], [1, 0]])
            mask = tf.constant([0, 2, 1])
            self.assertAllEqual(sess.run(updator(state, mask)),
                                [[1, 0], [1, 0], [0, 1]])

            updator = spin_swap_updator_onehot.SpinSwapUpdatorOneHot((2, 2, 3))
            state = tf.constant([[[1, 0, 0], [1, 0, 0]],
                                 [[0, 1, 0], [0, 0, 1]]])
            mask = tf.constant([0, 2, 1, 3])
            self.assertAllEqual(sess.run(updator(state, mask)),
                                [[[1, 0, 0], [0, 1, 0]],
                                 [[1, 0, 0], [0, 0, 1]]])


if __name__ == "__main__":
    tf.test.main()
