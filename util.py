"""Util functions for neural quantum states."""
import os
import pickle
from typing import Text, Optional

import tensorflow as tf


def get_models_dir(output_dir: Text) -> Text:
    return os.path.join(output_dir, 'models')


def get_params_dir(output_dir: Text) -> Text:
    return os.path.join(output_dir, 'params')


def get_latest_state_path(output_dir: Text) -> Text:
    return os.path.join(output_dir, 'latest_state.npy')


def get_latest_model_prefix(output_dir: Text) -> Text:
    return os.path.join(output_dir, 'models', 'latest_model.ckpt')


def get_params_prefix(output_dir: Text) -> Text:
    return os.path.join(output_dir, 'params', 'params.pickle')


class Logger(object):
    """A logger that can log train information to screen or file."""

    def __init__(self, log_file: Text) -> None:
        self._log_file = log_file

    def write_to_all(self, *args, **kwargs) -> None:
        """Writes log to screen and file."""
        print(*args, **kwargs)
        with open(self._log_file, 'a') as file:
            print(file=file, *args, **kwargs)

    def write_to_file(self, *args, **kwargs) -> None:
        """Writes log to file only."""
        with open(self._log_file, 'a') as file:
            print(file=file, *args, **kwargs)


def save_params_to_pickle_file(session: tf.Session,
                               params_filename: Text) -> None:
    """Saves all trainable parameters to a python pickle file.

    Args:
        session: A Tensorflow session contains the model parameters.
        params_filename: The file name where model parameters are saved to.
    """
    params = {}
    for var in tf.trainable_variables():
        params[var.name] = var.eval(session=session)
    with open(params_filename, 'wb') as f:
        pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)


def load_params_from_pickle_file(session: tf.Session,
                                 params_filename: Text) -> None:
    """Loads trainable parameters from a python pickle file.

    Args:
        session: A Tensorflow session contains the model parameters.
        params_filename: The file name where model parameters are loaded from.
    """
    with open(params_filename, 'rb') as f:
        params = pickle.load(f)
        for var in tf.trainable_variables():
            session.run(var.assign(params[var.name]))


def save_model(session: tf.Session, model_dir: Text, global_step: int = None,
               max_to_keep: int = 5) -> Text:
    """Saves model to a directory.

    Args:
        session: A Tensorflow session contains the model.
        model_dir: The model directory to save the model.
        global_step: The global step in training.
        max_to_keep: Max number of models to keep in the saved model directory.
    """
    saver = tf.train.Saver(max_to_keep=max_to_keep)
    return saver.save(session, model_dir, global_step=global_step)


def load_model(session: tf.Session, model_dir: Text) -> None:
    """Loads model from a directory.

    Args:
        session: A Tensorflow session contains the model.
        model_dir: The model directory from which to load the model.
    """
    saver = tf.train.Saver()
    saver.restore(session, model_dir)


def get_step_from_checkpoint_path(checkpoint_path: Text) -> Optional[int]:
    """Gets the current step from checkpoint path.

    Args:
        checkpoint_path: The checkpoint path string, could be a model checkpoint
            or a parameters checkpoint.

    Returns: The current step in the checkpoint path. Return None if step does
        not exist in the checkpoint path.

    """
    if not checkpoint_path:
        return None
    try:
        return int(checkpoint_path.strip().split("-")[-1])
    except ValueError:
        return None
