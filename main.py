"""Main methods to solve 1D SU(N) systems using off-policy VMC algorithm."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Tuple

import numpy as np
import tensorflow as tf

import mcmc_sampler
import networks
import nqs_solver
import off_policy_runner
import util
from operators import heisenberg2d_square
from operators import states_util
from operators import spin_loop_correlator
from operators import sun_spin1d
from updators import spin_swap_updator_onehot
from updators import spin_swap_updator_value

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", "",
                    "The output directory.")

flags.DEFINE_string("system", "sun_spin1d",
                    "The system to study. One of 'sun_spin1d' or "
                    "'heisenberg2d_square'.")

flags.DEFINE_string("mode", None,
                    "Mode for `train` or `measure`. The `train` mode trains "
                    "the network. The `measure` mode measures physical "
                    "quantities.")

# Logging frequency parameters.
flags.DEFINE_integer("n_print", 10,
                     "Write and print logs every `n_print` iteration steps.")
flags.DEFINE_integer("n_print_optimization", 1,
                     "Write and print logs every `n_print_optimization` steps "
                     "in the off-policy optimization.")
flags.DEFINE_integer("n_save", 10,
                     "Save model and parameters every `n_save` iteration "
                     "steps.")

# Optimization parameters.
flags.DEFINE_integer("n_iter", 1500,
                     "Number of total iterations.")
flags.DEFINE_integer("n_sample", 20000,
                     "Number of sampled states in each iteration.")
flags.DEFINE_integer("n_sample_initial", 1000, "The initial `n_sample`.")
flags.DEFINE_integer("n_sample_warmup", 10,
                     "The warmup steps for `n_sample`. We use a small "
                     "`n_sample` at the beginning to avoid out-of-memory "
                     "issue.")
flags.DEFINE_integer("n_optimize_1", 100,
                     "Number of off-policy optimization steps in each iteration"
                     " in the first stage.")
flags.DEFINE_integer("n_optimize_2", 10,
                     "Number of off-policy optimization steps in each iteration"
                     " in the second stage.")
flags.DEFINE_integer("change_n_optimize_at", None,
                     "The iteration step at which to change the number of "
                     "off-policy optimization steps.")

flags.DEFINE_float("learning_rate_1", 1e-3,
                   "The learning rate in the first stage.")
flags.DEFINE_float("learning_rate_2", 1e-4,
                   "The learning rate in the second stage.")
flags.DEFINE_integer("change_learning_rate_at", None,
                     "The iteration step at which to change the learning rate.")

flags.DEFINE_string("load_param_from", None,
                    "load the parameters from a dictionary pickle file,the keys"
                    " are the names of the parameters, the values are the numpy"
                    " array of the parameters.")
flags.DEFINE_string("load_model_from", None,
                    "load the model from tensorflow model checkpoint.")
flags.DEFINE_string("load_state_from", None,
                    "load the state from file.")

# Flags for SU(N) systems parameters.
flags.DEFINE_integer("n_sites", 10, "Number of spin sites.")
flags.DEFINE_integer("n_spin", 2, "Number of spin components.")
flags.DEFINE_integer("layers", 3, "Number of convolutional layers.")
flags.DEFINE_integer("filters", 8, "Number of filters.")
flags.DEFINE_integer("kernel", 3, "Size of kernel.")
flags.DEFINE_string("network", "CNN", "Network architecture.")
flags.DEFINE_string("state_encoding", "one_hot",
                    "The format for state encoding, `value` or `one_hot`")
flags.DEFINE_string("dtype", "tf.float32",
                    "Data type for tensorflow model.")

# Flags for measuring loop permutation correlation operators.
flags.DEFINE_integer("origin_site_index", 0, "The origin site index.")
flags.DEFINE_integer("sigma_1", None,
                     "The goal spin index in the local SU(N) generator. If "
                     "None, the loop permutation operator will not select a "
                     "goal spin index.")
flags.DEFINE_integer("sigma_2", None,
                     "The start spin index in the local SU(N) generator. If "
                     "None, the loop permutation operator will not select a"
                     "start spin index.")
flags.DEFINE_bool("add_fermi_sign", True,
                  "Whether or not to add back Fermi sign.")
flags.DEFINE_integer("n_batch", 5000,
                     "The size of a batch of states input into the network each "
                     "time for measurement using sampling.")


def validate_flags():
    if not FLAGS.output_dir:
        raise ValueError("Output dir must be specified.")

    if FLAGS.n_sites % FLAGS.n_spin != 0:
        raise ValueError("In order to have a spin balanced chain, the number "
                         "of sites {} must be divided by number of spin "
                         "components {}".format(FLAGS.nsites, FLAGS.n_spin))


def initialize_heisenberg2d_square(
        sess: tf.Session, logger: util.Logger, writer: tf.summary.FileWriter,
        dtype: tf.DType) -> Tuple[heisenberg2d_square.Heisenberg2DSquare,
                                  mcmc_sampler.MCMCSampler,
                                  off_policy_runner.OffPolicyRunner,
                                  nqs_solver.NqsSolver,
                                  networks.Network]:
    """Spin balanced 2D Heisenberg AFM model on square lattice.

    We consider a square geometry, i.e. n_sites_x = n_sites_y
    """
    if FLAGS.n_spin != 2:
        raise ValueError(
            "Heisenberg AFM model spin other than 1/2 not implemented.")
    n_sites = (FLAGS.n_sites, FLAGS.n_sites)
    n_sites_total = int(np.prod(n_sites))
    spin_nums = [n_sites_total // FLAGS.n_spin] * FLAGS.n_spin

    hamiltonian = heisenberg2d_square.Heisenberg2DSquare(True)

    if FLAGS.state_encoding == 'value':
        ini_state = states_util.random_state2d(n_sites, FLAGS.n_spin, spin_nums,
                                               False)
        updator = spin_swap_updator_value.SpinSwapUpdatorValue(ini_state.shape)
    elif FLAGS.state_encoding == 'one_hot':
        ini_state = states_util.random_state2d(n_sites, FLAGS.n_spin, spin_nums,
                                               True)
        updator = spin_swap_updator_onehot.SpinSwapUpdatorOneHot(
            ini_state.shape)
    else:
        raise NotImplementedError('state encoding {} not implemented'.
                                  format(FLAGS.state_encoding))

    if FLAGS.network == 'CNN':
        # The layer architecture is similar to https://arxiv.org/abs/1903.06713
        # The filter shape, the first i=hidden layer activation are different.
        filters = [12, 10, 8, 6, 4, 2]

        # Initialize the wavefunction network for MCMC sampling.
        mh_net = networks.SpinNetConv2D(n_sites, FLAGS.n_spin,
                                        layers=len(filters),
                                        filters=filters,
                                        kernel=FLAGS.kernel,
                                        state_encoding=FLAGS.state_encoding,
                                        data_format='channels_last')

        # Initialize the wavefunction network for off-policy optimization.
        # Which is the same as and shares parameters with the sampling network.
        op_net = networks.SpinNetConv2D(n_sites, FLAGS.n_spin,
                                        layers=len(filters),
                                        filters=filters,
                                        kernel=FLAGS.kernel,
                                        state_encoding=FLAGS.state_encoding)

    else:
        raise NotImplementedError(
            'network {} for heisenberg2d square not implemented'.format(
                FLAGS.network))
    mhsampler = mcmc_sampler.MCMCSampler(ini_state, mh_net, updator, sess,
                                         dtype=dtype, logger=logger)
    oprunner = off_policy_runner.OffPolicyRunner(n_sites_total, ini_state.shape,
                                                 op_net, sess, dtype=dtype,
                                                 logger=logger, writer=writer)
    solver = nqs_solver.NqsSolver(n_sites_total, hamiltonian, mhsampler,
                                  oprunner, logger)

    return hamiltonian, mhsampler, oprunner, solver, mh_net


def initialize_sun_spin1d(
        sess: tf.Session, logger: util.Logger, writer: tf.summary.FileWriter,
        dtype: tf.DType) -> Tuple[sun_spin1d.SUNSpin1D,
                                  mcmc_sampler.MCMCSampler,
                                  off_policy_runner.OffPolicyRunner,
                                  nqs_solver.NqsSolver,
                                  networks.Network]:
    """1D SU(N) Spin model with equal spins on each spin component."""
    spin_nums = [FLAGS.n_sites // FLAGS.n_spin] * FLAGS.n_spin

    hamiltonian = sun_spin1d.SUNSpin1D(t=1.0, pbc=True)

    if FLAGS.state_encoding == 'value':
        ini_state = states_util.random_state1d(FLAGS.n_sites, FLAGS.n_spin,
                                               spin_nums, False)
        updator = spin_swap_updator_value.SpinSwapUpdatorValue(ini_state.shape)
    elif FLAGS.state_encoding == 'one_hot':
        ini_state = states_util.random_state1d(FLAGS.n_sites, FLAGS.n_spin,
                                               spin_nums, True)
        updator = spin_swap_updator_onehot.SpinSwapUpdatorOneHot(
            ini_state.shape)
    else:
        raise NotImplementedError('state encoding {} not implemented'.
                                  format(FLAGS.state_encoding))

    if FLAGS.network == 'CNN':
        # Initialize the wavefunction network for MCMC sampling.
        mh_net = networks.SpinNetConv1D(FLAGS.n_sites, FLAGS.n_spin,
                                        layers=FLAGS.layers,
                                        filters=[FLAGS.filters] * FLAGS.layers,
                                        kernel=FLAGS.kernel,
                                        local_params=None,
                                        state_encoding=FLAGS.state_encoding,
                                        data_format='channels_last')

        # Initialize the wavefunction network for off-policy optimization.
        # Which is the same as and shares parameters with the sampling network.
        op_net = networks.SpinNetConv1D(FLAGS.n_sites, FLAGS.n_spin,
                                        layers=FLAGS.layers,
                                        filters=[FLAGS.filters] * FLAGS.layers,
                                        kernel=FLAGS.kernel,
                                        local_params=None,
                                        state_encoding=FLAGS.state_encoding)
    elif FLAGS.network == 'RBM':
        # Initialize the wavefunction network for MCMC sampling.
        mh_net = networks.RestrictedBoltzmannMachine1D(
            FLAGS.n_sites, FLAGS.n_spin, filters=FLAGS.filters,
            kernel=FLAGS.kernel, state_encoding=FLAGS.state_encoding,
            data_format='channels_last')

        # Initialize the wavefunction network for off-policy optimization.
        # Which is the same as and shares parameters with the sampling network.
        op_net = networks.RestrictedBoltzmannMachine1D(
            FLAGS.n_sites, FLAGS.n_spin, filters=FLAGS.filters,
            kernel=FLAGS.kernel, state_encoding=FLAGS.state_encoding)
    else:
        raise NotImplementedError('network {} for sun spin1d not implemented'.
                                  format(FLAGS.network))
    mhsampler = mcmc_sampler.MCMCSampler(ini_state, mh_net, updator, sess,
                                         dtype=dtype, logger=logger)
    oprunner = off_policy_runner.OffPolicyRunner(FLAGS.n_sites, ini_state.shape,
                                                 op_net, sess, dtype=dtype,
                                                 logger=logger, writer=writer)
    solver = nqs_solver.NqsSolver(FLAGS.n_sites, hamiltonian, mhsampler,
                                  oprunner, logger)

    return hamiltonian, mhsampler, oprunner, solver, mh_net


def _write_parameters_log_info(logger):
    logger.write_to_all('{:<30} {:<30}'.format("output_dir", FLAGS.output_dir))
    logger.write_to_all('{:<30} {:<30}'.format("system", FLAGS.system))
    logger.write_to_all('{:<30} {:<30}'.format("mode", FLAGS.mode))
    logger.write_to_all('{:<30} {:<30}'.format("n_print", FLAGS.n_print))
    logger.write_to_all('{:<30} {:<30}'.format("n_print_optimization",
                                               FLAGS.n_print_optimization))
    logger.write_to_all('{:<30} {:<30}'.format("n_save", FLAGS.n_save))
    logger.write_to_all('{:<30} {:<30}'.format("n_iter", FLAGS.n_iter))
    logger.write_to_all('{:<30} {:<30}'.format("n_sample", FLAGS.n_sample))
    logger.write_to_all(
        '{:<30} {:<30}'.format("n_sample_initial", FLAGS.n_sample_initial))
    logger.write_to_all(
        '{:<30} {:<30}'.format("n_sample_warmup", FLAGS.n_sample_warmup))
    logger.write_to_all(
        '{:<30} {:<30}'.format("n_optimize", FLAGS.n_optimize_1))
    if FLAGS.change_n_optimize_at is not None:
        logger.write_to_all('at step {}, change n_optimize to {}'.format(
            FLAGS.change_n_optimize_at, FLAGS.n_optimize_2))
    else:
        FLAGS.change_n_optimize_at = FLAGS.n_iter + 1
    logger.write_to_all(
        '{:<30} {:<30}'.format("learning rate", FLAGS.learning_rate_1))
    if FLAGS.change_learning_rate_at is not None:
        logger.write_to_all('at step {}, change learning rate to {}'.format(
            FLAGS.change_learning_rate_at, FLAGS.learning_rate_2))
    else:
        FLAGS.change_learning_rate_at = FLAGS.n_iter + 1
    logger.write_to_all("Parameters for system:")
    logger.write_to_all(
        '{:<30} {:<30}'.format("Number of sites:", FLAGS.n_sites))
    logger.write_to_all(
        '{:<30} {:<30}'.format("Number of spin components:", FLAGS.n_spin))
    logger.write_to_all(
        '{:<30} {:<30}'.format("Number of layers:", FLAGS.layers))
    logger.write_to_all(
        '{:<30} {:<30}'.format("Number of filters:", FLAGS.filters))
    logger.write_to_all('{:<30} {:<30}'.format("Kernel size:", FLAGS.kernel))
    logger.write_to_all(
        '{:<30} {:<30}'.format("Network architecture:", FLAGS.network))
    logger.write_to_all(
        '{:<30} {:<30}'.format("State encoding:", FLAGS.state_encoding))
    logger.write_to_all('{:<30} {:<30}'.format("network dtype", FLAGS.dtype))

    if FLAGS.mode == 'energy':
        logger.write_to_all("Parameters for measuring energy:")
        logger.write_to_all(
            '{:<30} {:<30}'.format("n_batch", FLAGS.n_batch))

    if FLAGS.mode == 'loop_correlator':
        logger.write_to_all(
            "Parameters for measuring loop permutation operators:")
        logger.write_to_all("S^{a, b}_{i, j}, where j = i, i + 1, ...")
        logger.write_to_all(
            '{:<30} {:<30}'.format("origin site index i",
                                   FLAGS.origin_site_index))
        logger.write_to_all(
            '{:<30} {:<30}'.format("spin index a", str(FLAGS.sigma_1)))
        logger.write_to_all(
            '{:<30} {:<30}'.format("spin index b", str(FLAGS.sigma_2)))
        logger.write_to_all(
            '{:<30} {:<30}'.format("Add back fermi sign:",
                                   str(FLAGS.add_fermi_sign)))
        logger.write_to_all(
            "If a and b are all None, the loop permutation operator "
            "doesn't depend on spin.")


def n_sample_fn(step: int) -> int:
    """`n_sample` scheduling function.

    Linearly increase `n_sample` in the first `n_sample_warmup` steps.
    """
    n_sample_increment = (
            (FLAGS.n_sample - FLAGS.n_sample_initial) // FLAGS.n_sample_warmup)
    if step < FLAGS.n_sample_warmup:
        return FLAGS.n_sample_initial + n_sample_increment * step
    else:
        return FLAGS.n_sample


def n_optimize_fn(step: int) -> int:
    """`n_optimize` scheduling function."""
    if step <= FLAGS.change_n_optimize_at:
        return FLAGS.n_optimize_1
    else:
        return FLAGS.n_optimize_2


def learning_rate_fn(step: int) -> int:
    """`learning_rate` scheduling function."""
    if step <= FLAGS.change_learning_rate_at:
        return FLAGS.learning_rate_1
    else:
        return FLAGS.learning_rate_2


def _calculate_global_step(current_step: int) -> int:
    """Calculate the current global step given the current iteration step."""
    global_step = 0
    for step in range(current_step):
        global_step += n_optimize_fn(step)
    return global_step


def _restore_calculation(session: tf.Session,
                         mhsampler: mcmc_sampler.MCMCSampler,
                         oprunner: off_policy_runner.OffPolicyRunner,
                         solver: nqs_solver.NqsSolver,
                         logger: util.Logger) -> bool:
    """Restores calculation from failures.

    If there are latest checkpoints in the output dir, it means there is a
    failure that leads to the previous calculation stopped.

    Args:
        session: The Tensorflow session.
        mhsampler: The Metropolis-Hasting MCMC sampler.
        oprunner: The Off-Policy runner.
        solver: The Nqs solver.
        logger: A Logger that prints to file and screen.

    Returns:
        True if the calculation is successfully restored.
    """
    latest_checkpoint = tf.train.latest_checkpoint(
        util.get_models_dir(FLAGS.output_dir))
    if latest_checkpoint is None:
        return False
    logger.write_to_all(
        "Found latest checkpoint {}, trying to restore calculation.".format(
            latest_checkpoint))
    current_step = util.get_step_from_checkpoint_path(latest_checkpoint) + 1
    if current_step is None:
        logger.write_to_all(
            "Failed to get current step from latest checkpoint.")
        return False
    latest_state_path = util.get_latest_state_path(FLAGS.output_dir)
    if not tf.gfile.Exists(latest_state_path):
        logger.write_to_all("Failed to find latest state.")
        return False

    util.load_model(session, latest_checkpoint)
    logger.write_to_all('Model loaded from {}'.format(latest_checkpoint))
    with tf.io.gfile.GFile(latest_state_path, "rb") as f:
        mhsampler.state = np.load(f)
        logger.write_to_all('State loaded from {}'.format(
            latest_state_path))
    solver.start_step = current_step
    oprunner.global_step = _calculate_global_step(current_step) + 1
    return True


def _save_measured_values(name, values, step: int = None,
                          logger: util.Logger = None) -> None:
    """Saves measured values to file.

    Args:
        name: The name for the measured values.
        values: A numpy.ndarray of the measured values.
        step: The current checkpoint step for the measurement. If the step is
            present, the measured values are saved to a numpy file with '-step'
            after the name.
        logger: A Logger that prints to file and screen.
    """
    energies_dir = os.path.join(FLAGS.output_dir, "measured_{}".format(name))
    os.makedirs(energies_dir, exist_ok=True)
    if step is None:
        energies_path = os.path.join(energies_dir, name)
    else:
        energies_path = os.path.join(energies_dir, "{}-{}".format(name, step))
    np.save(energies_path, values)
    if logger is not None:
        logger.write_to_all(
            '{} saved to path {}.npy'.format(name, energies_path))


def main():
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    os.makedirs(util.get_models_dir(FLAGS.output_dir), exist_ok=True)
    os.makedirs(util.get_params_dir(FLAGS.output_dir), exist_ok=True)

    # A logger to write logs to screen and file.
    logger = util.Logger(os.path.join(FLAGS.output_dir, 'log.txt'))

    # A writer to save quantities to tensorboard.
    writer = tf.summary.FileWriter(FLAGS.output_dir)

    _write_parameters_log_info(logger)

    if FLAGS.dtype == 'tf.float32':
        dtype = tf.float32
    elif FLAGS.dtype == 'tf.float64':
        dtype = tf.float64
    else:
        raise NotImplementedError(
            'network dtype {} not implemented'.format(FLAGS.dtype))

    # Create a tf session.
    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1)
    sess = tf.Session(config=session_conf)

    # Create hamiltonian, sampler, runner, solver, and build tf graphs.
    if FLAGS.system == 'sun_spin1d':
        hamiltonian, mhsampler, oprunner, solver, net = (
            initialize_sun_spin1d(sess, logger, writer, dtype))
    elif FLAGS.system == 'heisenberg2d_square':
        hamiltonian, mhsampler, oprunner, solver, net = (
            initialize_heisenberg2d_square(sess, logger, writer, dtype))
    else:
        raise ValueError("System {} not implemented.".format(FLAGS.system))

    if FLAGS.mode != 'train' or not _restore_calculation(
            sess, mhsampler, oprunner, solver, logger):
        if FLAGS.load_model_from is not None:
            util.load_model(sess, FLAGS.load_model_from)
            logger.write_to_all('nqs model loaded from {}'.format(
                FLAGS.load_model_from))
        elif FLAGS.load_param_from is not None:
            util.load_params_from_pickle_file(sess, FLAGS.load_param_from)
            logger.write_to_all('nqs parameters loaded from {}'.format(
                FLAGS.load_param_from))
        else:
            sess.run(tf.global_variables_initializer())
            logger.write_to_all('nqs parameters randomly initialized.')

        if FLAGS.load_state_from is not None:
            mhsampler.state = np.load(FLAGS.load_state_from)
            logger.write_to_all('state loaded from {}'.format(
                FLAGS.load_state_from))

    writer.add_graph(sess.graph)
    writer.flush()

    # Start the computation.
    if FLAGS.mode == 'train':
        solver.train(n_iter=FLAGS.n_iter,
                     n_sample_fn=n_sample_fn,
                     n_optimize_fn=n_optimize_fn,
                     n_print=FLAGS.n_print,
                     n_print_optimization=FLAGS.n_print_optimization,
                     n_save=FLAGS.n_save,
                     learning_rate_fn=learning_rate_fn,
                     output_dir=FLAGS.output_dir)
    elif FLAGS.mode == 'energy':
        values = solver.sample(FLAGS.n_sample, FLAGS.n_batch, [hamiltonian])
        measure_name = "energies"
        logger.write_to_all(
            "Sampled {} states, the average energy: {}, variance: {}.".format(
                values.shape[1], values.mean(),
                (values ** 2.0).mean() - values.mean() ** 2.0))
        _save_measured_values(measure_name, values[0], logger=logger)

        # If the step in the checkpoint path is present, save measured values
        # to a file with step in the name.
        step = util.get_step_from_checkpoint_path(FLAGS.load_param_from)
        if step is not None:
            _save_measured_values(measure_name, values[0], step, logger)
    elif FLAGS.mode == 'loop_correlator':
        correlators = []
        i = FLAGS.origin_site_index
        for j in range(i, i + FLAGS.n_sites):
            correlators.append(spin_loop_correlator.SpinLoopCorrelator(
                i, j % FLAGS.n_sites, FLAGS.sigma_1, FLAGS.sigma_2,
                add_sign=FLAGS.add_fermi_sign))
        values = solver.sample(FLAGS.n_sample, FLAGS.n_batch, correlators)
        if FLAGS.sigma_1 and FLAGS.sigma_2:
            measure_name = "loop_correlators_i{}".format(i)
            loop_correlator_str = "S^{}{}({}...j)".format(FLAGS.sigma_1,
                                                          FLAGS.siamg_2, i)
        else:
            measure_name = "loop_correlators_i{}".format(i)
            loop_correlator_str = "({}...j)".format(i)
        logger.write_to_all(
            "Sampled {} states, the average loop correlators <{}> "
            "(j = {}, {}, ... {}):\n {}\n variances:\n {}.".format(
                values.shape[1], loop_correlator_str, i, i + 1,
                i - 1 if i > 0 else FLAGS.n_sites - 1, values.mean(axis=1),
                ((values ** 2.0).mean(axis=1) - values.mean(axis=1) ** 2.0)))
        _save_measured_values(measure_name, values, logger=logger)

        # If the step in the checkpoint path is present, save measured values
        # to a file with step in the name.
        step = util.get_step_from_checkpoint_path(FLAGS.load_param_from)
        if step is not None:
            _save_measured_values(measure_name, values, step, logger)
    else:
        raise NotImplementedError('mode {} not implemented.'.format(FLAGS.mode))


if __name__ == '__main__':
    main()
