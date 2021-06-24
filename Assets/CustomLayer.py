# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy
import types as python_types
import warnings

from keras import initializers
from keras import regularizers
from keras import constraints
from keras import activations
from keras import backend as K
from keras.engine.base_layer import InputSpec
from keras.engine.base_layer import Layer
from keras.utils.generic_utils import func_dump
from keras.utils.generic_utils import func_load
from keras.utils.generic_utils import deserialize_keras_object
from keras.utils.generic_utils import has_arg
from keras.utils.generic_utils import get_custom_objects


class NoisyDense(Layer):
    def __init__(self, units,
                 sigma_init=0.017,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='lecun_uniform', #'he_normal',
                 bias_initializer='lecun_uniform', #'he_normal',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 sigma_kernel_initializer=None,
                 sigma_bias_initializer=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units
        self.sigma_init = sigma_init
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.sigma_kernel_initializer = initializers.Constant(value=self.sigma_init)
        self.sigma_bias_initializer = initializers.Constant(value=self.sigma_init)

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True
        # self.input_dim = None
        get_custom_objects().update({'NoisyDense': NoisyDense})

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True)
        if self.sigma_init:
            self.sigma_kernel = self.add_weight(shape=(self.input_dim, self.units),
                                        initializer=self.sigma_kernel_initializer,
                                        name='sigma_kernel',
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        trainable=True)
        self.epsilon_kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      initializer=initializers.RandomNormal(mean=0.0, stddev=1.0),
                                      name='epsilon_kernel',
                                      regularizer=None,
                                      constraint=None,
                                      trainable=False)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=True)
            if self.sigma_init:
                self.sigma_bias = self.add_weight(shape=(self.units,),
                                            initializer=self.sigma_bias_initializer,
                                            name='sigma_bias',
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint,
                                            trainable=True)
            self.epsilon_bias = self.add_weight(shape=(self.units,),
                                        initializer=initializers.RandomNormal(mean=0.0, stddev=1.0),
                                        name='epsilon_bias',
                                        regularizer=None,
                                        constraint=None,
                                        trainable=False)
        else:
            self.bias = None
            self.sigma_bias = None
            self.epsilon_bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: self.input_dim})       
        self.built = True

    def call(self, inputs):
        kernel = self.kernel + self.sigma_kernel * self.epsilon_kernel if self.sigma_init else self.kernel + self.epsilon_kernel
        output = K.dot(inputs, kernel)
        if self.use_bias:
            bias = self.bias + self.sigma_bias * self.epsilon_bias if self.sigma_init else self.bias + self.epsilon_bias
            output = K.bias_add(output, bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'sigma_init': self.sigma_init,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'sigma_kernel_initializer': initializers.serialize(self.sigma_kernel_initializer),
            'sigma_bias_initializer': initializers.serialize(self.sigma_bias_initializer),
        }
        base_config = super(NoisyDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LayerNormalization(Layer):
    def __init__(self,
                 center=True,
                 scale=True,
                 epsilon=None,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 **kwargs):
        """Layer normalization layer

        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)

        :param center: Add an offset parameter if it is True.
        :param scale: Add a scale parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        :param gamma_initializer: Initializer for the gamma weight.
        :param beta_initializer: Initializer for the beta weight.
        :param gamma_regularizer: Optional regularizer for the gamma weight.
        :param beta_regularizer: Optional regularizer for the beta weight.
        :param gamma_constraint: Optional constraint for the gamma weight.
        :param beta_constraint: Optional constraint for the beta weight.
        :param kwargs:

        Rearranged CyberZHG's Implimentation.
        https://github.com/CyberZHG/keras-layer-normalization
        
        MIT License
        """
        super(LayerNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.center = center
        self.scale = scale
        if epsilon is None:
            epsilon = K.epsilon() * K.epsilon()
        self.epsilon = epsilon
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma, self.beta = None, None

    def build(self, input_shape):
        shape = input_shape[-1:]
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                name='gamma',
            )
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                name='beta',
            )
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_constraint': constraints.serialize(self.gamma_constraint),
            'beta_constraint': constraints.serialize(self.beta_constraint),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))