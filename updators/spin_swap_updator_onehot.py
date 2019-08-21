"""Swap updator for spin state with one-hot encoding."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple

import numpy as np
import tensorflow as tf

from updators import updator


class SpinSwapUpdatorOneHot(updator.Updator):

    def __init__(self, shape: Tuple[int, ...]) -> None:
        """Initializes swap updator for spin state with one-hot encoding.

        The dimension of spin state is arbitrary.

        Args:
            shape: The shape of the spin state. The input rank must be larger or
                equal to 2.
        """
        if len(shape) < 2:
            raise ValueError(
                "Invalid shape {} for one-hot encoded states".format(shape))
        self._shape = shape
        self._n_sites = int(np.prod(self._shape[:-1]))
        self._n_spin = self._shape[-1]

    def masks_placeholder(self) -> tf.Tensor:
        return tf.placeholder(dtype=tf.int32, shape=[None, self._n_sites],
                              name='Masks')

    def generate_masks(self, n_sample: int) -> np.ndarray:
        rang = range(n_sample)
        swaps = np.random.randint(0, self._n_sites, (2, n_sample))
        masks = np.arange(self._n_sites)[None, :].repeat(n_sample, axis=0)
        masks[rang, swaps[0]], masks[rang, swaps[1]] = (
            masks[rang, swaps[1]], masks[rang, swaps[0]])
        return masks

    def __call__(self, state: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        state = tf.reshape(state, [-1, self._n_spin])
        return tf.reshape(tf.gather(state, mask), self._shape)
