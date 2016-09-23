from engine import RNNCell, weight
from keras import initializations, regularizers, activations
from keras import backend as K
import numpy as np


class SimpleRNNCell(RNNCell):

	def __init__(self, output_dim, init='glorot_uniform', inner_init='orthogonal', activation='tanh', W_regularizer=None, U_regularizer=None, b_regularizer=None, **kwargs):
		self.output_dim = output_dim
		self.init = initializations.get(init)
		self.inner_init = initializations.get(inner_init)
		self.activation = activations.get(activation)
		self.W_regularizer = regularizers.get(W_regularizer)
		self.U_regularizer = regularizers.get(U_regularizer)
		self.b_regularizer = regularizers.get(b_regularizer)
		super(SimpleRNNCell, self).__init__(**kwargs)

	def build(self, input_shape):
		input_dim = input_shape[-1]
		h = (-1, self.output_dim)
		W = weight((input_dim, self.output_dim), init=self.init, regularizer=self.W_regularizer)
		U = weight((self.output_dim, self.output_dim), init=self.inner_init, regularizer=self.U_regularizer)
		b = weight((self.output_dim,), init='zero', regularizer=self.b_regularizer)

		def step(x, states, weights):
			h = states[0]
			W, U, b = weights
			h = self.activation(K.dot(x, W) + K.dot(h, U) + b)
			return h, [h]

		self.step = step
		self.states = [h]
		self.weights = [W, U, b]
		super(SimpleRNNCell, self).build(input_shape)

	def get_config(self):
		config = {'output_dim': self.output_dim,
				  'init': self.init.__name__,
				  'inner_init': self.inner_init.__name__,
				  'activation': self.activation.__name__,
				  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
				  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
				  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None}
		base_config = super(SimpleRNNCell, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class GRUCell(RNNCell):

	def __init__(self, output_dim, init='glorot_uniform', inner_init='orthogonal', activation='tanh', inner_activation='hard_sigmoid', W_regularizer=None, U_regularizer=None, b_regularizer=None, **kwargs):
		self.output_dim = output_dim
		self.init = initializations.get(init)
		self.inner_init = initializations.get(inner_init)
		self.activation = activations.get(activation)
		self.inner_activation = activations.get(inner_activation)
		self.W_regularizer = regularizers.get(W_regularizer)
		self.U_regularizer = regularizers.get(U_regularizer)
		self.b_regularizer = regularizers.get(b_regularizer)
		super(GRUCell, self).__init__(**kwargs)

	def build(self, input_shape):
		input_dim = input_shape[-1]
		h = (-1, self.output_dim)
		W = weight((input_dim, 3 * self.output_dim,), init=self.init, regularizer=self.W_regularizer)
		U = weight((self.output_dim, 3 * self.output_dim,), init=self.inner_init, regularizer=self.U_regularizer)
		b = weight((3 * self.output_dim,), init='zero', regularizer=self.b_regularizer)

		def step(x, states, weights):
			h_tm1 = states[0]
			W, U, b = weights
			matrix_x = K.dot(x, W) + b
			matrix_inner = K.dot(h_tm1, U[:, :2 * self.output_dim])
			x_z = matrix_x[:, :self.output_dim]
			x_r = matrix_x[:, self.output_dim: 2 * self.output_dim]
			inner_z = matrix_inner[:, :self.output_dim]
			inner_r = matrix_inner[:, self.output_dim: 2 * self.output_dim]
			z = self.inner_activation(x_z + inner_z)
			r = self.inner_activation(x_r + inner_r)
			x_h = matrix_x[:, 2 * self.output_dim:]
			inner_h = K.dot(r * h_tm1, U[:, 2 * self.output_dim:])
			hh = self.activation(x_h + inner_h)
			h = z * h_tm1 + (1 - z) * hh
			return h, [h]

		self.step = step
		self.states = [h]
		self.weights = [W, U, b]
		super(GRUCell, self).build(input_shape)

	def get_config(self):
		config = {'output_dim': self.output_dim,
				  'init': self.init.__name__,
				  'inner_init': self.inner_init.__name__,
				  'activation': self.activation.__name__,
				  'inner_activation': self.inner_activation.__name__,
				  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
				  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
				  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None}
		base_config = super(GRUCell, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class LSTMCell(RNNCell):

	def __init__(self, output_dim, init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one', activation='tanh', inner_activation='hard_sigmoid', W_regularizer=None, U_regularizer=None, b_regularizer=None, **kwargs):
		self.output_dim = output_dim
		self.init = initializations.get(init)
		self.inner_init = initializations.get(inner_init)
		self.forget_bias_init = initializations.get(forget_bias_init)
		self.activation = activations.get(activation)
		self.inner_activation = activations.get(inner_activation)
		self.W_regularizer = regularizers.get(W_regularizer)
		self.U_regularizer = regularizers.get(U_regularizer)
		self.b_regularizer = regularizers.get(b_regularizer)
		super(LSTMCell, self).__init__(**kwargs)

	def build(self, input_shape):
		input_dim = input_shape[-1]
		W = weight((input_dim, 4 * self.output_dim,), init=self.init, regularizer=self.W_regularizer)
		U = weight((self.output_dim, 4 * self.output_dim,), init=self.inner_init, regularizer=self.U_regularizer)
		b = np.concatenate([np.zeros(self.output_dim), K.get_value(self.forget_bias_init((self.output_dim,))), np.zeros(2 * self.output_dim)])
		b = weight(b, regularizer=self.b_regularizer)
		h = (-1, self.output_dim)
		c = (-1, self.output_dim)

		def step(x, states, weights):
			h_tm1, c_tm1 = states
			W, U, b = weights
			z = K.dot(x, W) + K.dot(h_tm1, U) + b
			z0 = z[:, :self.output_dim]
			z1 = z[:, self.output_dim: 2 * self.output_dim]
			z2 = z[:, 2 * self.output_dim: 3 * self.output_dim]
			z3 = z[:, 3 * self.output_dim:]
			i = self.inner_activation(z0)
			f = self.inner_activation(z1)
			c = f * c_tm1 + i * self.activation(z2)
			o = self.inner_activation(z3)
			h = o * self.activation(c)
			return h, [h, c]

		self.step = step
		self.states = [h, c]
		self.weights = [W, U, b]
		super(LSTMCell, self).build(input_shape)

	def get_config(self):
		config = {'output_dim': self.output_dim,
				  'init': self.init.__name__,
				  'inner_init': self.inner_init.__name__,
				  'forget_bias_init': self.forget_bias_init.__name__,
				  'activation': self.activation.__name__,
				  'inner_activation': self.inner_activation.__name__,
				  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
				  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
				  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None}
		base_config = super(LSTMCell, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
