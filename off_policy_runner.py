"""Off-Policy algorithm runner for optimization and measurement."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import sys
import time
from typing import Callable, Tuple, Optional

import numpy as np
import tensorflow as tf

import util

_OPNodes = collections.namedtuple("OPNodes",
                                  ["State", "LnPhi", "Count", "LnPhi0",
                                   "UState", "UCoeffs", "Lr", "op", "O", "Os",
                                   "WsNorm"])


class OffPolicyRunner(object):

    def __init__(self, n_sites: int, shape: Tuple[Optional[int], ...],
                 lnphi_net: Callable[[tf.Tensor], tf.Tensor],
                 sess: tf.Session, dtype: tf.DType = tf.float32,
                 logger: util.Logger = None,
                 writer: tf.summary.FileWriter = None):
        """Initializes an Off-Policy algorithm runner.

        This class can perform Off-Policy optimization and also calculate the
        local values of observables （i.e. doing measurements).

        This class also doesn't initialize variable parameters.

        Args:
            n_sites: Number of spin sites.
            shape: The shape of the states.
            lnphi_net: A function that construct the lnphi net. given an input
                tensor x of shape (..., *shape), lnphi_net(x) output the result
                tensor of shape (..., 1) representing the log of the
                wavefunction.
            sess: The Tensorflow session that contains the graph.
            dtype: he data type for the network parameters.
            logger: A Logger that prints to file and screen.
            writer: TF summary FileWriter.
        """

        self._n_sites = n_sites
        self._shape = shape
        self._dtype = dtype
        self._logger = logger
        self._writer = writer

        self._lnphi_net = lnphi_net

        self._Nodes = self._build_graph()

        self._summary = tf.summary.merge_all()

        self._sess = sess
        self._global_step = 0

    @property
    def global_step(self) -> int:
        """The global step in the Off-Policy runner."""
        return self._global_step

    @global_step.setter
    def global_step(self, global_step: int) -> None:
        """Sets the global step in the Off-Policy runner."""
        self._global_step = global_step

    @property
    def sess(self) -> tf.Session:
        return self._sess

    def _build_graph(self) -> _OPNodes:
        """Builds the Off-Policy network graph."""

        with tf.name_scope('Off-Policy'):
            # The placeholder for the MCMC sampled states. The shape is
            # (batch size, shape of state).
            State = tf.placeholder(dtype=self._dtype,
                                   shape=(None,) + self._shape, name='State')

            # The ln(phi) values of the MCMC sampled states. The network
            # parameters to compute ln(phi) are the current parameters. The
            # shape is (batch size,)
            LnPhi = self._lnphi_net(State)

            # The Counts for every sampled states. Since we can only consider
            # unique sampled states to reduce the memory usage. The shape is
            # (batch size,)
            Count = tf.placeholder(dtype=self._dtype, shape=(None,),
                                   name='Count')
            tf.summary.scalar('n_unique', tf.shape(Count)[0])

            # The placeholder for the initial ln(phi) values of the MCMC sampled
            # states. The shape is (batch size,)
            LnPhi0 = tf.placeholder(dtype=self._dtype, shape=(None,),
                                    name='LnPhi0')

            # The placeholder for the updated states by the Hamiltonian. The
            # shape is (batch size, num of updates, shape of state). The
            # num of updates is usually linear in system size for a locally
            # interacting hamiltonian.
            UState = tf.placeholder(dtype=self._dtype,
                                    shape=(None, None) + self._shape,
                                    name='UState')

            # The ln(phi) values for the updated states. The network parameters
            # to compute ln(phi) are the current parameters. The shape is
            # (batch size, num of updates,)
            ULnPhi = self._lnphi_net(UState)

            # The placeholder for the Hamiltonian coefficients for the updated
            # states. The shape is (batch size, num of updates).
            UCoeffs = tf.placeholder(dtype=self._dtype, shape=(None, None),
                                     name='UCoeffs')

            # The importance sampling (IS) weights. The shape is (batch size,).
            with tf.name_scope('Is'):
                DLnPhi = LnPhi - LnPhi0

                # This is to avoid the exponential explosion.
                DLnPhi = DLnPhi - tf.reduce_mean(DLnPhi, keepdims=True)
                ratio = tf.exp(DLnPhi * 2.0)
                Ws = Count * ratio

                # We need to normalize the IS weights since the wavefunction is
                # not normalized.
                WsNorm = tf.reduce_sum(Ws, name='normalization')
                Ws = tf.stop_gradient(Ws / WsNorm)

            # The local energies under the current network parameters. The shape
            # is (batch size,). When doing measurement, the local energies can
            # be substituted to any other local quantities for any other
            # operators.
            with tf.name_scope('Es'):
                DULnPhi = ULnPhi - self._lnphi_net(State)[..., None]
                Os = tf.stop_gradient(
                    tf.reduce_sum(UCoeffs * tf.exp(DULnPhi), axis=1))

            # The total energy, which is a scalar.
            with tf.name_scope('E'):
                O = tf.stop_gradient(tf.reduce_sum(Ws * Os))
                tf.summary.scalar('E', O)

            # The gradient stopped loss.
            with tf.name_scope('Loss'):
                # Note that the gradient can only flow back through LnPhi.
                Loss = tf.reduce_sum(Ws * Os * LnPhi) - O * tf.reduce_sum(
                    Ws * LnPhi)
                tf.summary.scalar('Loss', Loss)

            # The placeholder for the adjustable learning rate.
            Lr = tf.placeholder(dtype=self._dtype, shape=[], name='Lr')

            # The operation to optimize the network parameters.
            op = tf.train.AdamOptimizer(Lr).minimize(Loss)

        return _OPNodes(State, LnPhi, Count, LnPhi0, UState, UCoeffs, Lr, op, O,
                        Os, WsNorm)

    def off_policy(self, n_optimize: int, n_print: int, states: np.ndarray,
                   lnphis: np.ndarray, counts: np.ndarray,
                   update_states: np.ndarray, update_coeffs: np.ndarray,
                   learning_rate: float) -> None:
        """Optimizes parameters for NQS using the off-Policy optimization.

        Args:
            n_optimize: The number of optimization iterations.
            n_print: The number of iteration steps for print.
            states: The sampled states, shape (batch size, shape of state).
            lnphis: The log wavefunction values for the sampled states,
                shape (num of samples,).
            counts: The counts of states in sampling, shape (batch size,).
            update_states: The states generated by the Hamiltonian for all the
                sampled states. The shape is (batch size, num of updates,
                shape of state), where num of updates is the max number of
                generated states among all sampled states. For a state with
                updated states fewer than num of updates, padding are used.
            update_coeffs: The Hamiltonian coefficients for the generated
                states, shape (batch size, num of updates).
            learning_rate: The initial learning rate for Adam optimizer.
        """
        feed_dict = {self._Nodes.State: states,
                     self._Nodes.Count: counts,
                     self._Nodes.LnPhi0: lnphis,
                     self._Nodes.UState: update_states,
                     self._Nodes.UCoeffs: update_coeffs,
                     self._Nodes.Lr: learning_rate}

        tic = time.time()
        for i in range(n_optimize):
            self._sess.run(self._Nodes.op,
                           feed_dict=feed_dict)

            if i % n_print == 0 and self._logger is not None:
                try:
                    E, WN, summary = self._sess.run(
                        [self._Nodes.O, self._Nodes.WsNorm, self._summary],
                        feed_dict=feed_dict)
                    self._logger.write_to_file(
                        "off-policy i: {:<5d} E: {:<20.16f} E/N: {:<20.16f} "
                        "n_unique: {:<6d} weight normalization: {:<20.16f} "
                        "time: {:.6f}".format(i, E, E / self._n_sites,
                                              states.shape[0],
                                              WN / counts.sum(),
                                              time.time() - tic))
                except Exception as e:
                    self._logger.write_to_all('error occurred!')
                    self._logger.write_to_all(e)
                    sys.exit(e)
                tic = time.time()
                if np.isnan(E):
                    self._logger.write_to_all(
                        'algorithm diverge, energy becomes nan.')
                    sys.exit('algorithm diverge, energy becomes nan.')
                self._writer.add_summary(summary, self._global_step)
                self._writer.flush()
            self._global_step += 1

    def measure(self, states: np.ndarray, update_states: np.ndarray,
                update_coeffs: np.ndarray) -> np.ndarray:
        """Calculates local values of an observable for the sampled states.

        M is the number of samples, shape is the shape tuple of the system.

        Args:
            states: The sampled states, shape (batch size, shape of state).
            update_states: The states generated by the operator for all the
                sampled states, shape (batch size, num of updates,
                shape of state), where P is max number of generated states among
                all sampled states.
            update_coeffs: The operator coefficients for the generated states,
                shape (batch size, num of updates).

        Returns:
            The values of an observable for the sampled states, shape
            (batch size,).

        """
        return self._sess.run(self._Nodes.Os,
                              feed_dict={self._Nodes.State: states,
                                         self._Nodes.UState: update_states,
                                         self._Nodes.UCoeffs: update_coeffs})
