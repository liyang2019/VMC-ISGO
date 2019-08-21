"""Functions for building backend networks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from typing import List, Text, Tuple, Sequence

import numpy as np
import tensorflow as tf


def conv_periodic_padding(inputs: tf.Tensor, filters: int, kernel_size: int,
                          strides: int,
                          data_format: Text, dimension: Text,
                          use_bias: bool = True, name: Text = None):
    """Periodic padding for convolution with periodic boundary conditions.

    The padding is consistent and is based only on `kernel_size`, not on the
    dimensions of `inputs` (as opposed to using `tf.layers.conv1d` alone).

    Args:
        inputs: A tensor of shape [batch, channels, width_in] or
            [batch, width_in, channels] depending on data_format for
            dimension = '1d' size [batch, channels, width_in, height_in] or
            [batch, width_in, height_in, channels] depending on data_format for
            dimension = '2d'.
        filters: The number of filters.
        kernel_size: The kernel size.
        strides: The stride size.
        data_format: The input format ('channels_last' or 'channels_first').
        dimension: The input dimension ('1d' or '2d').
        use_bias: Whether or not use bias in the convolutional layer.
        name: The name of the convolutional layer.

    Returns:
        A tensor with the same format as the input with the data either intact
        (if kernel_size == 1) or padded (if kernel_size > 1).
    """

    inputs = periodic_padding(inputs, kernel_size, data_format, dimension)
    if dimension == '1d':
        return tf.layers.conv1d(
            inputs=inputs, filters=filters, kernel_size=kernel_size,
            strides=strides,
            padding='VALID', use_bias=use_bias,
            data_format=data_format, name=name, reuse=tf.AUTO_REUSE)
    else:
        return tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=kernel_size,
            strides=strides,
            padding='VALID', use_bias=use_bias,
            data_format=data_format, name=name, reuse=tf.AUTO_REUSE)


def periodic_padding(inputs: tf.Tensor, kernel_size: int, data_format: Text,
                     dimension: Text):
    """Pads the inputs with periodic bounding condition.

    Args:
        inputs: A tensor of shape [batch, channels, width_in] or
            [batch, width_in, channels] depending on data_format for
            dimension = '1d' size [batch, channels, width_in, height_in] or
            [batch, width_in, height_in, channels] depending on data_format for
            dimension = '2d'
        kernel_size: The kernel size.
        data_format: The input format ('channels_last' or 'channels_first').
        dimension: The input dimension ('1d' or '2d').

    Returns:
        A tensor with the same format as the input with the data either intact
        (if kernel_size == 1) or padded (if kernel_size > 1).

    Raises:
        ValueError: If the dimension is not in ['1d', '2d'].
    """
    if dimension == '1d':
        if data_format == 'channels_first':
            padded_inputs = tf.concat((inputs, inputs[:, :, 0:kernel_size - 1]),
                                      axis=2)
        else:
            padded_inputs = tf.concat((inputs, inputs[:, 0:kernel_size - 1, :]),
                                      axis=1)
    elif dimension == '2d':
        if data_format == 'channels_first':
            padded_inputs = tf.concat(
                (inputs, inputs[:, :, 0:kernel_size - 1, :]), axis=2)
            padded_inputs = tf.concat(
                (padded_inputs, padded_inputs[:, :, :, 0:kernel_size - 1]),
                axis=3)
        else:
            padded_inputs = tf.concat(
                (inputs, inputs[:, 0:kernel_size - 1, :, :]), axis=1)
            padded_inputs = tf.concat(
                (padded_inputs, padded_inputs[:, :, 0:kernel_size - 1, :]),
                axis=2)
    else:
        raise ValueError('The dimension you want has not been implemented yet.')

    return padded_inputs


def _process_inputs_tensor(inputs: tf.Tensor, n_spin: int,
                           state_encoding: Text, data_format: Text,
                           dimension: Text):
    """Processes the inputs Tensor according to state encoding and data format.

    Args:
        inputs: The inputs tensor, if using value state encoding, it is of shape
            (..., n_sites), if using one-hot state encoding, it is of shape
            (..., n_sites, n_spin)
        n_spin: Number of spin components.
        state_encoding:  The format for state encoding, 'value' or
            'one_hot'.
        data_format: Input format ('channels_last', 'channels_first', or
            None). If set to None, the format is dependent on whether a GPU
            is available.
        dimension: The input dimension ('1d' or '2d').

    Returns:
        (a) The processed inputs tensor with shape
            (batch_size, n_sites, channel)
        (b) The original batch shape. The batch_size in the output tensor equals
            to prod(batch_shape)

    Raises:
        ValueError:
            (a) If the state encoding is not in ['value', 'one_hot'].
            (b) If the dimension is not in ['1d', '2d'].

    """
    inputs_shape = tf.shape(inputs)
    if dimension == '1d':
        if state_encoding == 'value':
            n_sites = inputs_shape[-1]
            batch_shape = inputs_shape[:-1]
            scale = tf.constant(2.0 / (n_spin - 1), dtype=inputs.dtype)
            inputs = scale * inputs - 1.0
            inputs = tf.reshape(inputs, (-1, n_sites, 1))
        elif state_encoding == 'one_hot':
            n_sites = inputs_shape[-2]
            batch_shape = inputs_shape[:-2]
            inputs = tf.reshape(inputs, (-1, n_sites, n_spin))
        else:
            raise ValueError(
                'Encoding {} not implemented'.format(state_encoding))
        if data_format == 'channels_first':
            inputs = tf.transpose(inputs, [0, 2, 1])
    elif dimension == '2d':
        if state_encoding == 'value':
            n_sites = inputs_shape[-2:]
            batch_shape = inputs_shape[:-2]
            scale = tf.constant(2.0 / (n_spin - 1), dtype=inputs.dtype)
            inputs = scale * inputs - 1.0
            inputs = tf.reshape(inputs, (-1, n_sites[0], n_sites[1], 1))
        elif state_encoding == 'one_hot':
            n_sites = inputs_shape[-3:-1]
            batch_shape = inputs_shape[:-3]
            inputs = tf.reshape(inputs, (-1, n_sites[0], n_sites[1], n_spin))
        else:
            raise ValueError(
                'Encoding {} not implemented'.format(state_encoding))
        if data_format == 'channels_first':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])
    else:
        raise ValueError('The dimension you want has not been implemented yet.')
    return inputs, batch_shape


def _process_local_fields(local_fields: np.ndarray, filters: int, kernel: int,
                          data_format):
    """Adds local fields to inputs according to data format.

    Args:
        filters: Number of filters.
        kernel: Kernel size.
        local_fields: The local fields.
        data_format: Input format ('channels_last', 'channels_first', or
            None). If set to None, the format is dependent on whether a GPU
            is available.

    Returns:
        A tensor of shape (batch_size, n_sites, channel) representing the
        encoding of the local fields.
    """
    local_field = np.concatenate(
        [local_fields,
         local_fields[:kernel - 1]])
    if data_format == 'channels_last':
        local_field = local_field[None, :, None]
    else:
        local_field = local_field[None, None, :]
    local_field = tf.constant(local_field)
    local_field = tf.layers.conv1d(
        local_field,
        filters=filters,
        kernel_size=kernel,
        data_format=data_format,
        activation=None,
        use_bias=False,
        reuse=tf.AUTO_REUSE,
        name='hiddenl')
    return local_field


class Network(abc.ABC):

    def __init__(self):
        """Initialize a Network."""

    @abc.abstractmethod
    def __call__(self, inputs: tf.Tensor):
        """Returns the output tensor of the network acting on the input tensor.

        The input tensor should have the shape of [..., *shape], where "..." are
        arbitrary dimensions, and shape is the shape of the state.

        Args:
            inputs: The input tensor.

        Returns: The output tensor of the network acting on the input tensor.

        """

    def feature_maps(self) -> List[tf.Tensor]:
        """Returns the feature maps of all convolutional layers."""
        return []


class RestrictedBoltzmannMachine1D(Network):

    def __init__(self, n_sites: int, n_spin: int, filters: int, kernel: int,
                 state_encoding: Text = 'value', data_format: Text = None):
        """Initializes an 1D Restricted Boltzmann Machine(RBM) network with PBC.

        The RBM network is a 1-hidden layer NN with PBC followed by a reduce sum
        along both the spatial and channel.

        The activation function for RBM is log(2cosh(x)) = log(e^{x} + e^{-x})

        Args:
            n_sites: Number of spin sites.
            n_spin: Number of spin components.
            filters: Number of filters.
            kernel: Kernel size.
            state_encoding:  The format for state encoding, 'value' or
                'one_hot'.
            data_format: Input format ('channels_last', 'channels_first', or
                None). If set to None, the format is dependent on whether a GPU
                is available.
        """
        super(RestrictedBoltzmannMachine1D, self).__init__()
        self._n_sites = n_sites
        self._n_spin = n_spin
        self._filters = filters
        self._kernel = kernel
        self._state_encoding = state_encoding
        if data_format:
            self._data_format = data_format
        elif tf.test.is_built_with_cuda():
            self._data_format = 'channels_first'
        else:
            self._data_format = 'channels_last'
        self._state_encoding = state_encoding

    def __call__(self, inputs):
        """Constructs the lnphi network.

        Args:
            inputs: The input state tensor, should have size [..., n_sites].

        Returns:
            The output lnphi tensor, should have size [...].
        """
        with tf.name_scope('RBM1D'):
            inputs, batch_shape = _process_inputs_tensor(
                inputs, self._n_spin, self._state_encoding,
                self._data_format, '1d')
            inputs = conv_periodic_padding(
                inputs, self._filters, self._kernel, 1, self._data_format,
                '1d', use_bias=True, name='hidden0')

            # The activation function for RBM is log(2cosh(x)), which can
            # be implemented by using softplus function: log(e^x + 1):
            #               log(2cosh(x)) = softplus(2x) - x
            inputs = tf.nn.softplus(inputs * 2.0) - inputs
            inputs = tf.reduce_sum(inputs, axis=[1, 2])
            return tf.reshape(inputs, batch_shape)


class SpinNetConv1D(Network):

    def __init__(self, n_sites: int, n_spin: int, layers: int,
                 filters: Sequence[int], kernel: int,
                 local_params: np.ndarray = None,
                 state_encoding: Text = 'value', data_format: Text = None):
        """Initializes an 1D CNN with pbc.

        Args:
            n_sites: Number of spin sites.
            n_spin: Number of spin components.
            layers: Number of hidden layers.
            filters: Number of filters.
            kernel: Kernel size.
            local_params: Local parameters for an inhomogeneous model.
            state_encoding: The format for state encoding, 'value' or
                'one_hot'.
            data_format: Input format ('channels_last', 'channels_first', or
                None). If set to None, the format is dependent on whether a GPU
                is available.

        Raises:
            ValueError:
                (a) If the number of layers is not larger than zero.
                (b) If the local parameters is present but not one-dimensional.
                (c) The size of filters is not the same as the number of layers.
        """
        super(SpinNetConv1D, self).__init__()
        if layers <= 0:
            raise ValueError(
                "Number of layers {} must be larger than 1.".format(layers))
        if local_params is not None and local_params.ndim != 1:
            raise ValueError(
                "`local_params` must be one-dimensional, however it is {} "
                "dimensional.".format(local_params.ndim))
        if len(filters) != layers:
            raise ValueError(
                "The size of filters {} does not equal to the number of layers "
                "{}.".format(len(filters), layers))
        self._n_sites = n_sites
        self._n_spin = n_spin
        self._layers = layers
        self._filters = filters
        self._kernel = kernel
        self._local_params = local_params
        self._state_encoding = state_encoding
        if data_format:
            self._data_format = data_format
        elif tf.test.is_built_with_cuda():
            self._data_format = 'channels_first'
        else:
            self._data_format = 'channels_last'
        self._state_encoding = state_encoding

        self._feature_maps = []

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        """Constructs the lnphi network.

        Args:
            inputs: The input state tensor, should have size (..., n_sites).

        Returns:
            The output lnphi tensor, should have size (...).
        """
        self._feature_maps.clear()
        with tf.name_scope('SpinNetConv1D'):
            inputs, batch_shape = _process_inputs_tensor(inputs, self._n_spin,
                                                         self._state_encoding,
                                                         self._data_format,
                                                         '1d')

            inputs = conv_periodic_padding(inputs, self._filters[0],
                                           self._kernel,
                                           1, self._data_format, '1d',
                                           use_bias=True, name='hidden0')
            if self._local_params is not None:
                inputs += _process_local_fields(self._local_params,
                                                self._filters[0], self._kernel,
                                                self._data_format)
            inputs = tf.nn.relu(inputs)
            self._feature_maps.append(inputs)
            for l in range(1, self._layers):
                inputs = conv_periodic_padding(inputs, self._filters[l],
                                               self._kernel, 1,
                                               self._data_format, '1d',
                                               use_bias=True,
                                               name='hidden' + str(l))
                inputs = tf.nn.relu(inputs)
                self._feature_maps.append(inputs)
            cvo = tf.reduce_sum(
                inputs, axis=1 if self._data_format == 'channels_last' else 2)
            cvo = tf.layers.dense(cvo, units=1,
                                  activation=None,
                                  use_bias=False,
                                  reuse=tf.AUTO_REUSE,
                                  name='output')
            return tf.reshape(cvo, batch_shape)

    def feature_maps(self) -> List[tf.Tensor]:
        """Returns the feature maps of all convolutional layers."""
        return self._feature_maps


class SpinNetConv2D(Network):

    def __init__(self, n_sites: Tuple[int, int], n_spin: int, layers: int,
                 filters: Sequence[int], kernel: int,
                 state_encoding: Text = 'value',
                 data_format: Text = None):
        """Initializes a 2D CNN with pbc.

        Args:
            n_sites: Number of spin sites along x and y directions.
            n_spin: Number of spin components.
            layers: Number of hidden layers.
            filters: Number of filters for each hidden layer.
            kernel: Kernel size.
            state_encoding:
            state_encoding: The format for state encoding, 'value' or
                'one_hot'.
            data_format: Input format ('channels_last', 'channels_first', or
                None). If set to None, the format is dependent on whether a GPU
                is available.

        Raises:
            ValueError:
                (a) If number of layers is not larger than 0.
                (b) The size of filters is not the same as the number of layers.
        """
        super(SpinNetConv2D, self).__init__()
        if layers <= 0:
            raise ValueError(
                "Number of layers {} must be larger than 0.".format(layers))
        if len(filters) != layers:
            raise ValueError(
                "The size of filters {} does not equal to the number of layers "
                "{}.".format(len(filters), layers))
        self._n_sites = n_sites
        self._n_spin = n_spin
        self._layers = layers
        self._filters = filters
        self._kernel = kernel
        self._state_encoding = state_encoding
        if data_format:
            self._data_format = data_format
        elif tf.test.is_built_with_cuda():
            self._data_format = 'channels_first'
        else:
            self._data_format = 'channels_last'
        self._state_encoding = state_encoding

        self._feature_maps = []

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        """Constructs the lnphi network.

        Args:
            inputs: The input state tensor, should have size
                (..., n_sites[0], n_sites[1]).

        Returns:
            The output lnphi tensor, should have size (...).
        """
        self._feature_maps.clear()
        with tf.name_scope('SpinNetConv2D'):
            inputs, batch_shape = _process_inputs_tensor(inputs, self._n_spin,
                                                         self._state_encoding,
                                                         self._data_format,
                                                         '2d')
            for l in range(self._layers):
                inputs = conv_periodic_padding(inputs, self._filters[l],
                                               self._kernel, 1,
                                               self._data_format, '2d',
                                               use_bias=True,
                                               name='hidden' + str(l))
                inputs = tf.nn.relu(inputs)
                self._feature_maps.append(inputs)
            cvo = tf.reduce_sum(
                inputs,
                axis=[1, 2] if self._data_format == 'channels_last' else [2, 3])
            cvo = tf.layers.dense(cvo, units=1,
                                  activation=None,
                                  use_bias=False,
                                  reuse=tf.AUTO_REUSE,
                                  name='output')
            return tf.reshape(cvo, batch_shape)
