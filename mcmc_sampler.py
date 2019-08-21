"""Markov-Chain Monte Carlo sampler using Metropolis-Hasting algorithm."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import time
from typing import Callable, Tuple

import numpy as np
import tensorflow as tf

import util
from updators import updator

_MHNodes = collections.namedtuple("MHNodes",
                                  ["State0", "LnPhi0", "States", "Lnphis",
                                   "NSample", "Rands", "Masks"])


class MCMCSampler(object):
    """Markov-Chain Monte Carlo sampler using Metropolis-Hasting algorithm."""

    def __init__(self, initial_state: np.ndarray,
                 lnphi_net: Callable[[tf.Tensor], tf.Tensor],
                 updator: updator.Updator, sess: tf.Session,
                 dtype: tf.DType = tf.float32, logger: util.Logger = None):
        """Initialize a Metropolis-Hasting sampler using tensorflow graph.

        This class doesn't initialize variable parameters.

        Args:
            initial_state: The initial state.
            lnphi_net: A function that constructs the lnphi net. Given an input
                tensor x of shape (any shape, shape of state), lnphi_net(x)
                outputs the result tensor of shape (any shape,) representing the
                ln(phi).
            updator: A updator for updating the state.
            sess: The Tensorflow session in which the graph will be built.
            dtype: The data type for the network.
            logger: A Logger to write logs to screen and file.
        """

        self._shape = initial_state.shape
        self._lnphi_net = lnphi_net
        self._updator = updator
        self._dtype = dtype
        self._logger = logger

        # For sampling states, CPU may have better performance.
        with tf.device('/cpu:0'):
            self._Nodes = self._build_graph()

        self._sess = sess

        self._state = initial_state

    @property
    def sess(self) -> tf.Session:
        return self._sess

    @property
    def state(self) -> np.ndarray:
        return self._state

    @state.setter
    def state(self, state: np.ndarray) -> None:
        """Sets the state in the MCMC sampler.

        Args:
            state: The state to set.

        Raises:
            ValueError: If the state to set has different shape from the shape
                of the state in the MCMC sampler.
        """
        if state.shape != self._shape:
            raise ValueError(
                " If the state to set has different shape {} from the shape {} "
                "of the state in the MCMC sampler".format(state.shape,
                                                          self._shape))
        self._state = state

    def _build_graph(self) -> _MHNodes:
        """Builds the Metropolis-Hasting sampling computational graph.

        tf.while_loop is used to build the sampling graph.
        """
        with tf.name_scope('MH'):
            # The placeholder for the initial state for a run of MCMC sampling.
            State0 = tf.placeholder(dtype=self._dtype, shape=self._shape)

            # The ln(phi) value for the initial state.
            LnPhi0 = self._lnphi_net(State0)

            # The 'time step' in the MCMC sampling.
            t0 = tf.constant(0)

            # The placeholder for the number of samples to generate.
            n_sample = tf.placeholder(dtype=tf.int32, shape=[], name='NSample')

            # The TensorArray for storing the sampled states.
            state_ta0 = tf.TensorArray(dtype=self._dtype, size=n_sample,
                                       element_shape=tf.TensorShape(
                                           self._shape), name='StateTa0')

            # The TensorArray for storing the ln(phi) values for sampled states.
            lnphi_ta0 = tf.TensorArray(dtype=self._dtype, size=n_sample,
                                       element_shape=tf.TensorShape(()),
                                       name='LnPhiTa0')

            # The pre-generated random numbers for the MCMC algorithm.
            rands = tf.placeholder(dtype=self._dtype, shape=(None,),
                                   name='Rands')

            # The placeholder for the masks for updating the state.
            masks = self._updator.masks_placeholder()

            def _time_step(t, state_ta, lnphi_ta, state, lnphi):
                new_state = self._updator(state, masks[t])
                new_lnphi = self._lnphi_net(new_state[None, ...])[0]
                dlnphi = new_lnphi - lnphi

                new_state, new_lnphi = tf.cond(
                    tf.logical_or(dlnphi > 0, rands[t] <= tf.exp(dlnphi * 2.0)),
                    lambda: [new_state, new_lnphi],
                    lambda: [state, lnphi])
                state_ta = state_ta.write(t, new_state)
                lnphi_ta = lnphi_ta.write(t, new_lnphi)
                return t + 1, state_ta, lnphi_ta, new_state, new_lnphi

            _, state_final_ta, lnphi_final_ta, _, _ = tf.while_loop(
                cond=lambda i, *_: i < n_sample,
                body=_time_step,
                loop_vars=(t0, state_ta0, lnphi_ta0, State0, LnPhi0),
                back_prop=False)
            states_out = state_final_ta.stack(name='States')
            lnphis_out = lnphi_final_ta.stack(name='LnPhis')
        return _MHNodes(State0, LnPhi0, states_out, lnphis_out, n_sample, rands,
                        masks)

    def sample_states(self, n_sample: int) -> Tuple[np.ndarray, np.ndarray]:
        """Samples states with Metropolis–Hastings algorithm.

        Args:
            n_sample: The number of samples to generate.

        Returns:
            A Tuple of (sampled states, ln of wavefunction values)
        """
        tic = time.time()
        rands = np.random.rand(n_sample)
        masks = self._updator.generate_masks(n_sample)

        states, lnphis = self._sess.run(
            [self._Nodes.States, self._Nodes.Lnphis],
            feed_dict={self._Nodes.State0: self._state,
                       self._Nodes.Rands: rands,
                       self._Nodes.Masks: masks,
                       self._Nodes.NSample: n_sample})

        # After a batch of sampling, update the current state.
        self._state = states[-1].round().astype(int)
        if self._logger is not None:
            self._logger.write_to_file("time for sampling {} states: {}".format(
                n_sample, time.time() - tic))
        return states.round().astype(int), lnphis
