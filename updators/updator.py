"""Abstract class for Updators.

The updator class is used for updating states in the MCMC algorithm implemented
in Tensorflow.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf


class Updator(ABC):

    @abstractmethod
    def masks_placeholder(self) -> tf.Tensor:
        """Creates a placeholder for the masks for updating states."""

    @abstractmethod
    def generate_masks(self, n_sample: int) -> np.ndarray:
        """Generates `n_sample` masks for updating states in the Markov-Chain.

        `n_sample` random masks are generated, packed into a numpy array and feed
        into the Sampler graph. At every step of the MCMC sampling, the Updator
        network will take the current state and a mask to produce an updated
        state.

        Args:
            n_sample: Number of random masks generated.

        Returns:
            A numpy array of packed random masks.
        """

    @abstractmethod
    def __call__(self, state: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """Updates the state using the mask.

        At every step of the MCMC sampling, the Updator network will take the
        current state and a mask, and produce an updated state.

        Args:
            state: The current state to be updated.
            mask: The mask containing information about how to update the
            current state.
        Returns:
            The updated state.
        """
