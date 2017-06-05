from keras.models import Model
from keras import initializers
from keras import constraints
from keras import regularizers
from keras.layers import *
from .engine import RNNCell


def _slice(x, dim, index):
    return x[:, index * dim: dim * (index + 1)]


def get_slices(x, n):
    dim = int(K.int_shape(x)[1] / n)
    return [Lambda(_slice, arguments={'dim': dim, 'index': i}, output_shape=lambda s: (s[0], dim))(x) for i in range(n)]


class Identity(Layer):

    def call(self, x):
        return x + 0.


class ExtendedRNNCell(RNNCell):

    def __init__(self, units=None,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if units is None:
            assert 'output_dim' in kwargs, 'Missing argument: units'
        else:
            kwargs['output_dim'] = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        super(ExtendedRNNCell, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'activation': activations.serialize(self.activation),
            'recurrent_activation': activations.serialize(self.recurrent_activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(ExtendedRNNCell, self).get_config()
        config.update(base_config)
        return config


class SimpleRNNCell(ExtendedRNNCell):

    def build_model(self, input_shape):
        output_dim = self.output_dim
        output_shape = (input_shape[0], output_dim)
        x = Input(batch_shape=input_shape)
        h_tm1 = Input(batch_shape=output_shape)
        kernel = Dense(output_dim,
                       kernel_initializer=self.kernel_initializer,
                       kernel_regularizer=self.kernel_regularizer,
                       kernel_constraint=self.kernel_constraint,
                       use_bias=self.use_bias,
                       bias_initializer=self.bias_initializer,
                       bias_regularizer=self.bias_regularizer,
                       bias_constraint=self.bias_constraint)
        recurrent_kernel = Dense(output_dim,
                                 kernel_initializer=self.recurrent_initializer,
                                 kernel_regularizer=self.recurrent_regularizer,
                                 kernel_constraint=self.recurrent_constraint,
                                 use_bias=False)
        h = add([kernel(x), recurrent_kernel(h_tm1)])
        h = Activation(self.activation)(h)
        return Model([x, h_tm1], [h, Identity()(h)])


class GRUCell(ExtendedRNNCell):

    def build_model(self, input_shape):
        output_dim = self.output_dim
        input_dim = input_shape[-1]
        output_shape = (input_shape[0], output_dim)
        x = Input(batch_shape=input_shape)
        h_tm1 = Input(batch_shape=output_shape)
        kernel = Dense(output_dim * 3,
                       kernel_initializer=self.kernel_initializer,
                       kernel_regularizer=self.kernel_regularizer,
                       kernel_constraint=self.kernel_constraint,
                       use_bias=self.use_bias,
                       bias_initializer=self.bias_initializer,
                       bias_regularizer=self.bias_regularizer,
                       bias_constraint=self.bias_constraint)
        recurrent_kernel_1 = Dense(output_dim * 2,
                                   kernel_initializer=self.recurrent_initializer,
                                   kernel_regularizer=self.recurrent_regularizer,
                                   kernel_constraint=self.recurrent_constraint,
                                   use_bias=False)
        recurrent_kernel_2 = Dense(output_dim,
                                   kernel_initializer=self.recurrent_initializer,
                                   kernel_regularizer=self.recurrent_regularizer,
                                   kernel_constraint=self.recurrent_constraint,
                                   use_bias=False)
        kernel_out = kernel(x)
        recurrent_kernel_1_out = recurrent_kernel_1(h_tm1)
        x0, x1, x2 = get_slices(kernel_out, 3)
        r0, r1 = get_slices(recurrent_kernel_1_out, 2)
        z = add([x0, r0])
        z = Activation(self.recurrent_activation)(z)
        r = add([x1, r1])
        h_prime = add([recurrent_kernel_2(multiply([r, h_tm1])), x2])
        h_prime = Activation(self.activation)(h_prime)
        gate = Lambda(lambda x: x[0] * x[1] + (1. - x[0]) * x[2], output_shape=lambda s: s[0])
        h = gate([z, h_prime, h_tm1])
        return Model([x, h_tm1], [h, Identity()(h)])


class LSTMCell(ExtendedRNNCell):

    def build_model(self, input_shape):
        output_dim = self.output_dim
        input_dim = input_shape[-1]
        output_shape = (input_shape[0], output_dim)
        x = Input(batch_shape=input_shape)
        h_tm1 = Input(batch_shape=output_shape)
        c_tm1 = Input(batch_shape=output_shape)
        kernel = Dense(output_dim * 4,
                       kernel_initializer=self.kernel_initializer,
                       kernel_regularizer=self.kernel_regularizer,
                       kernel_constraint=self.kernel_constraint,
                       use_bias=self.use_bias,
                       bias_initializer=self.bias_initializer,
                       bias_regularizer=self.bias_regularizer,
                       bias_constraint=self.bias_constraint)
        recurrent_kernel = Dense(output_dim * 4,
                                 kernel_initializer=self.recurrent_initializer,
                                 kernel_regularizer=self.recurrent_regularizer,
                                 kernel_constraint=self.recurrent_constraint,
                                 use_bias=False)
        kernel_out = kernel(x)
        recurrent_kernel_out = recurrent_kernel(h_tm1)
        x0, x1, x2, x3 = get_slices(kernel_out, 4)
        r0, r1, r2, r3 = get_slices(recurrent_kernel_out, 4)
        f = add([x0, r0])
        f = Activation(self.recurrent_activation)(f)
        i = add([x1, r1])
        i = Activation(self.recurrent_activation)(i)
        c_prime = add([x2, r2])
        c_prime = Activation(self.activation)(c_prime)
        c = add([multiply([f, c_tm1]), multiply([i, c_prime])])
        c = Activation(self.activation)(c)
        o = add([x3, r3])
        h = multiply([o, c])
        return Model([x, h_tm1, c_tm1], [h, Identity()(h), c])
