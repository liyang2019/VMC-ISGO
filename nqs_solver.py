"""Neural Quantum State Solver using ISGO method.

The Important Sampling Gradient Optimization (ISGO) method is inspired from the
off-policy method in Reinforcement Learning (RL). The structure of the program
is similar to that of RL, so we use the name of Off-Policy in the program.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from typing import Callable, Sequence, Text, Tuple

import numpy as np
import tensorflow as tf

import mcmc_sampler
import off_policy_runner
import util
from operators import operator


def _get_unique_states(
        states: np.ndarray, lnphis: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns the unique states, their coefficients and the counts."""
    states, indices, counts = np.unique(states, return_index=True,
                                        return_counts=True, axis=0)
    lnphis = lnphis[indices]
    return states, lnphis, counts


def _generate_updates(
        states: np.ndarray,
        operator: operator.Operator) -> Tuple[np.ndarray, np.ndarray]:
    """Generates updated states and coefficients for an Operator.

    Args:
        states: The states with shape (batch size, shape of state).
        operator: The operator used for updating the states.

    Returns:
        The updated states and their coefficients. The shape of the updated
        states is (batch size, num of updates, shape of state), where num of
        updates is the largest number of updated states among all given states.
        If a state has fewer updated states, its updates are padded with the
        original state.

    """
    n_states = states.shape[0]
    ustates = np.empty(n_states, np.ndarray)
    ucoeffs = np.empty(n_states, np.ndarray)
    for i, state in enumerate(states):
        ustates[i], ucoeffs[i] = operator.find_states(state)
    lengths = np.array([uc.shape[0] for uc in ucoeffs])
    max_len = np.max(lengths)

    # Padding update states and coefficients.
    ustates_pad = states[:, None, :].repeat(max_len, 1)
    ucoeffs_pad = np.zeros((n_states, max_len))
    for i in range(n_states):
        ustates_pad[i, :lengths[i]] = ustates[i]
        ucoeffs_pad[i, :lengths[i]] = ucoeffs[i]

    return ustates_pad, ucoeffs_pad


class NqsSolver:

    def __init__(self, n_sites: int, hamiltonian: operator.Operator,
                 mhsampler: mcmc_sampler.MCMCSampler,
                 oprunner: off_policy_runner.OffPolicyRunner,
                 logger: util.Logger = None) -> None:
        """Initializes the Variational Monte Carlo Network Quantum State Solver.

        Args:
            n_sites: Total number of sites.
            hamiltonian: The Hamiltonian operator of the system.
            mhsampler: The Metropolis-Hasting Algorithm sampler.
            oprunner: The Off-Policy algorithm runner.
            logger: A Logger for print to file or screen.

        Raises:
            ValueError: If the MCMC sampler and the Off-Policy runner are using
                different Tensorflow sessions.
        """

        if mhsampler.sess != oprunner.sess:
            raise ValueError("The MCMC sampler and the Off-Policy runner must "
                             "be using the same Tensorflow session.")

        self._n_sites = n_sites
        self._hamiltonian = hamiltonian
        self._mhsampler = mhsampler
        self._oprunner = oprunner
        self._logger = logger
        self._sess = self._mhsampler.sess
        self._start_step = 0

    @property
    def start_step(self) -> int:
        """The start step of the solver for training."""
        return self._start_step

    @start_step.setter
    def start_step(self, start_step: int) -> None:
        """Sets the start step of the solver for training."""
        self._start_step = start_step

    def sample(self, n_sample: int, n_batch: int,
               operators: Sequence[operator.Operator]) -> np.ndarray:
        """Measures operators through sampling.

        Args:
            n_sample: The number of sampled states.
            n_batch: The size of a batch of states input into the network each
            time. This is to avoid out of memory issue.
            operators: A list of operators to be measured.

        Returns:
            The values of the operators for the sampled states, shape
            (n_operators, n_sample).

        """

        self._print_start_information()
        if self._logger is not None:
            self._logger.write_to_all('start measuring...')

        values = []
        while n_sample > n_batch:
            values.append(self._sample(n_batch, operators))
            n_sample -= n_batch
        values.append(self._sample(n_sample, operators))
        values = np.hstack(values)

        if self._logger is not None:
            self._logger.write_to_all(
                "The shape of the measured values numpy.ndarray: {}".format(
                    values.shape))
        return values

    def _sample(self, n_sample: int,
                operators: Sequence[operator.Operator]) -> np.ndarray:
        """Samples for a single batch."""
        tic = time.time()
        states, _ = self._mhsampler.sample_states(n_sample)
        unique_states = np.unique(states, axis=0)

        values = []
        for operator in operators:
            update_states, update_coeffs = _generate_updates(unique_states,
                                                             operator)

            unique_values = self._oprunner.measure(
                unique_states, update_states, update_coeffs)

            states_map = dict([(tuple(s.flatten()), v)
                               for s, v in zip(unique_states, unique_values)])
            values.append(np.empty(n_sample))
            for i, state in enumerate(states):
                values[-1][i] = states_map[tuple(state.flatten())]
        if self._logger is not None:
            self._logger.write_to_all(
                'time for measure: {}'.format(time.time() - tic))

        return np.vstack(values)

    def train(self, n_iter: int, n_sample_fn: Callable[[int], int],
              n_optimize_fn: Callable[[int], int], n_print: int,
              n_print_optimization: int, n_save: int,
              learning_rate_fn: Callable[[int], float],
              output_dir: Text) -> None:
        """Runs the Variational Monte Carlo loop with Off-Policy algorithm.

        Args:
            n_iter: The total number of iteration steps for training.
            n_sample_fn: The number of sampling in each iteration step.
            n_optimize_fn: The number of optimization (updating parameters) in
                each iteration step.
            n_print: The number of every iteration steps for printing.
            n_print_optimization: The number of every iteration steps in the
                Off-Policy runner for printing.
            n_save: The number of every iteration steps for saving the model.
            learning_rate_fn: The learning rate.
            output_dir: The output directory to save all the results.
        """
        if not os.path.exists(output_dir):
            raise ValueError("Output dir {} does not exist.".format(output_dir))

        self._print_start_information()
        if self._logger is not None:
            self._logger.write_to_all('start training...')

        tic = time.time()
        E_cum = 0.0  # The accumulated energy for print.
        n_cum = 0  # The accumulated iteration steps for print.
        for step in range(self._start_step, n_iter):

            n_sample = n_sample_fn(step)

            n_optimize = n_optimize_fn(step)

            states, lnphis = self._mhsampler.sample_states(n_sample)

            # Use unique states to reduce memory usage.
            states, lnphis, counts = _get_unique_states(states, lnphis)

            update_states, update_coeffs = _generate_updates(
                states, self._hamiltonian)

            n_unique = states.shape[0]

            Es = self._oprunner.measure(states, update_states,
                                        update_coeffs)

            E = np.sum(Es * counts) / n_sample
            E2 = np.sum((Es ** 2) * counts) / n_sample
            EVar = (E2 - E ** 2) / self._n_sites

            E_cum += E
            n_cum += 1
            if n_cum == n_print and self._logger is not None:
                E_av = E_cum / n_cum
                self._logger.write_to_all(
                    "step: {:<5d} E: {:<20.16f} E/N: {:<20.16f} "
                    "EVar: {:<20.16f} lr: {:<6.5f} n_sample: {:<6d} "
                    "n_unique: {:<6d} n_optimize: {:<5d} "
                    "time: {:.6f}".format(step, E_av, E_av / self._n_sites,
                                          EVar, learning_rate_fn(step),
                                          n_sample, n_unique, n_optimize,
                                          time.time() - tic))
                E_cum = 0.0
                n_cum = 0
                tic = time.time()

            self._oprunner.off_policy(
                n_optimize=n_optimize,
                n_print=n_print_optimization,
                states=states,
                lnphis=lnphis,
                counts=counts,
                update_states=update_states,
                update_coeffs=update_coeffs,
                learning_rate=learning_rate_fn(step))

            util.save_params_to_pickle_file(
                self._sess,
                util.get_params_prefix(output_dir) + '-{}'.format(step))

            if step % n_save == 0:
                self._save_checkpoint(output_dir, step)

        self._save_checkpoint(output_dir, n_iter)

    def _print_start_information(self):
        """Prints the state information for the computation."""
        if self._logger is not None:
            self._logger.write_to_all(
                'Initial state: \n{}'.format(self._mhsampler.state.T))
            n_param = 0
            for var in tf.trainable_variables():
                self._logger.write_to_all(var)
                n_param += np.prod(var.get_shape().as_list())
            self._logger.write_to_all('number of trainable parameters: {}'
                                      .format(int(n_param)))

    def _save_checkpoint(self, output_dir, step):
        """Saves current model checkpoint and state."""
        np.save(util.get_latest_state_path(output_dir),
                self._mhsampler.state)
        save_path = util.save_model(
            self._sess, util.get_latest_model_prefix(output_dir), step)
        if self._logger is not None:
            self._logger.write_to_all(
                "Most current model saved in path: {}".format(
                    save_path))
