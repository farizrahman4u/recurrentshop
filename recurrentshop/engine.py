from keras.layers import Layer, InputSpec
from keras.models import Sequential
from keras import initializations, regularizers
from keras.utils.layer_utils import layer_from_config
from keras import backend as K
from inspect import getargspec
import numpy as np
from . import backend


'''Provides a simpler API for building complex recurrent neural networks using Keras.

The RNN logic is written inside RNNCells, which are added sequentially to a RecurrentContainer.
A RecurrentContainer behaves similar to a Recurrent layer in Keras, and accepts arguments like
return_sequences, unroll, stateful, etc [See Keras Recurrent docstring]
The .add() method of a RecurrentContainer is used to add RNNCells and other layers to it. Each
element in the input sequence passes through the layers in the RecurrentContainer in the order
in which they were added.
'''


__author__ = "Fariz Rahman"
__copyright__ = "Copyright 2016, datalog.ai"
__credits__ = ["Fariz Rahman", "Malaikannan Sankarasubbu"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Fariz Rahman"
__email__ = "fariz@datalog.ai"
__status__ = "Production"


_backend = getattr(K, K.backend() + '_backend')


class learning_phase(object):

	def __init__(self, value):
		self.value = value

	def __enter__(self):
		self.learning_phase_place_holder = _backend._LEARNING_PHASE
		_backend._LEARNING_PHASE = self.value

	def __exit__(self, *args):
		_backend._LEARNING_PHASE = self.learning_phase_place_holder


if K.backend() == 'theano':
	rnn = backend.rnn
else:
	rnn = lambda *args, **kwargs: list(K.rnn(*args, **kwargs)) + [[]]


def _isRNN(layer):
	return issubclass(layer.__class__, RNNCell)

def _get_first_timestep(x):
	slices = [slice(None)] * K.ndim(x)
	slices[1] = 0
	return x[slices]

def _get_last_timestep(x):
	ndim = K.ndim(x)
	if K.backend() == 'tensorflow':
		import tensorflow as tf
		slice_begin = tf.pack([0, tf.shape(x)[1] - 1] + [0] * (ndim - 2))
		slice_size = tf.pack([-1, 1] + [-1] * (ndim - 2))
		last_output = tf.slice(x, slice_begin, slice_size)
		last_output = tf.squeeze(last_output, [1])
		return last_output
	else:
		return x[:, -1]


class weight(object):

	def __init__(self, value, init='glorot_uniform', regularizer=None, trainable=True, name=None):
		if type(value) == int:
			value = (value,)
		if type(value) in [tuple, list]:
			if type(init) == str:
				init = initializations.get(init)
			self.value = init(value, name=name)
		elif 'numpy' in str(type(value)):
			self.value = K.variable(value, name=name)
		else:
			self.value = value
		if type(regularizer) == str:
			regularizer = regularizers.get(regularizer)
		self.regularizer = regularizer
		self.trainable = trainable


class RNNCell(Layer):

	def __init__(self, **kwargs):
		if 'input_dim' in kwargs:
			kwargs['input_shape'] = (kwargs['input_dim'],)
			del kwargs['input_dim']
		if 'output_dim' in kwargs:
			self.output_dim = kwargs['output_dim']
			del kwargs['output_dim']
		self.initial_states = {}
		super(RNNCell, self).__init__(**kwargs)

	def _step(self, x, states):
		args = [x, states]
		if hasattr(self, 'weights'):
			args += [self.weights]
		if hasattr(self, 'constants'):
			args += [self.constants]
		args = args[:len(getargspec(self.step).args)]
		return self.step(*args)

	def build(self, input_shape):
		self.input_spec = [InputSpec(shape=input_shape)]
		super(RNNCell, self).build(input_shape)

	@property
	def weights(self):
		w = []
		if hasattr(self, 'trainable_weights'):
			w += self.trainable_weights
		if hasattr(self, 'non_trainable_weights'):
			w += self.non_trainable_weights
		return w

	def get_layer(self, **kwargs):
		rc = RecurrentContainer(**kwargs)
		rc.add(self)
		return rc

	@weights.setter
	def weights(self, ws):
		self.trainable_weights = []
		self.non_trainable_weights = []
		self.regularizers = []
		for w in ws:
			if not isinstance(w, weight):
				w = weight(w, name='{}_W'.format(self.name))
			if w.trainable:
				self.trainable_weights += [w.value]
			else:
				self.non_trainable_weights += [w.value]
			if w.regularizer:
				w.regularizer.set_param(w.value)
				self.regularizers += [w.regularizer]

	def get_output_shape_for(self, input_shape):
		if hasattr(self, 'output_dim'):
			return input_shape[:-1] + (self.output_dim,)
		else:
			return input_shape

	def get_config(self):
		config = {}
		if hasattr(self, 'output_dim'):
			config['output_dim'] = self.output_dim
		base_config = super(RNNCell, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class RecurrentContainer(Layer):

	def __init__(self, weights=None, return_sequences=False, go_backwards=False, stateful=False, readout=False, state_sync=False, decode=False, output_length=None, input_length=None, unroll=False, **kwargs):
		self.return_sequences = return_sequences or decode
		self.initial_weights = weights
		self.go_backwards = go_backwards
		self.stateful = stateful
		self.readout = readout
		self.state_sync = state_sync
		self.decode = decode
		if decode:
			assert output_length, 'Missing argument: output_length should be specified for decoders.'
		self.output_length = output_length
		self.input_length = input_length
		self.unroll = unroll
		if unroll:
			assert input_length, 'Missing argument: input_length should be specified for unrolling.'
		self.supports_masking = True
		self.model = Sequential()
		super(RecurrentContainer, self).__init__(**kwargs)

	def add(self, layer):
		'''Add a layer
		# Arguments:
		layer: Layer instance. RNNCell or a normal layer such as Dense.
		'''
		self.model.add(layer)
		self.uses_learning_phase = any([l.uses_learning_phase for l in self.model.layers])
		if len(self.model.layers) == 1:
			if layer.input_spec is not None:
				shape = layer.input_spec[0].shape
			else:
				shape = layer.input_shape
			if not self.decode:
				shape = (shape[0], self.input_length) + shape[1:]
			self.batch_input_shape = shape
			self.input_spec = [InputSpec(shape=shape)]
		if _isRNN(layer) and self.state_sync:
			if not hasattr(self, 'nb_states'):
				self.nb_states = len(layer.states)
			else:
				assert len(layer.states) == self.nb_states, 'Incompatible layer. In a state synchronized recurrent container, all the cells should have the same number of states.'
		if self.stateful:
			self.reset_states()

	def pop(self):
		'''Remove the last layer
		'''
		self.model.pop()
		if self.stateful:
			self.reset_states()

	@property
	def input_shape(self):
		return self.input_spec[0].shape

	@property
	def output_shape(self):
		shape = self.model.output_shape
		if self.decode:
			return (shape[0], self.output_length) + shape[1:]
		if self.return_sequences:
			input_length = self.input_spec[0].shape[1]
			return (shape[0], input_length) + shape[1:]
		else:
			return shape

	def get_output_shape_for(self, input_shape):
		# this is a container
		return self.output_shape

	def step(self, x, states):
		states = list(states)
		state_index = 0
		if self.decode:
			x = states[0]
			_x = x
			states = states[1:]
		for i in range(len(self.model.layers)):
			layer = self.model.layers[i]
			if self.readout and i == 0:
				if self.readout in ['add', True]:
					x += states[-1]
				elif self.readout == 'mul':
					x *= states[-1]
				elif self.readout == 'pack':
					x = K.pack([x, states[-1]])
				elif self.readout == 'readout_only':
					x = states[-1]
			if _isRNN(layer):
				if self.state_sync:
					x, new_states = layer._step(x, states[:len(layer.states)])
					states[:len(layer.states)] = new_states
				else:
					x, new_states = layer._step(x, states[state_index : state_index + len(layer.states)])
					states[state_index : state_index + len(layer.states)] = new_states
					state_index += len(layer.states)
			else:
				x = layer.call(x)
		if self.decode:
			states = [_x] + states
		if self.readout:
			states[-1] = x
		return x, states

	def call(self, x, mask=None):
		input_shape = self.input_spec[0].shape
		if self.stateful:
			initial_states = self.states
		else:
			initial_states = self.get_initial_states(x)
		if self.decode:
			initial_states = [x] + initial_states
			if self.uses_learning_phase:
				with learning_phase(0):
					last_output_0, outputs_0, states_0, updates = rnn(self.step, K.zeros((1, self.output_length, 1)), initial_states, unroll=self.unroll, input_length=self.output_length)
				with learning_phase(1):
					last_output_1, outputs_1, states_1, updates = rnn(self.step, K.zeros((1, self.output_length, 1)), initial_states, unroll=self.unroll, input_length=self.output_length)
				outputs = K.in_train_phase(outputs_1, outputs_0)
				last_output = _get_last_timestep(outputs)
				states = [K.in_train_phase(states_1[i], states_0[i]) for i in range(len(states_0))]
			else:
				last_output, outputs, states, updates = rnn(self.step, K.zeros((1, self.output_length, 1)), initial_states, unroll=self.unroll, input_length=self.output_length)
		else:
			if self.uses_learning_phase:
				with learning_phase(0):
					last_output_0, outputs_0, states_0, updates = rnn(self.step, x, initial_states, go_backwards=self.go_backwards, mask=mask, unroll=self.unroll, input_length=input_shape[1])
				with learning_phase(1):
					last_output_1, outputs_1, states_1, updates = rnn(self.step, x, initial_states, go_backwards=self.go_backwards, mask=mask, unroll=self.unroll, input_length=input_shape[1])
				outputs = K.in_train_phase(outputs_1, outputs_0)
				last_output = _get_last_timestep(outputs)
				states = [K.in_train_phase(states_1[i], states_0[i]) for i in range(len(states_0))]
			else:
				last_output, outputs, states, updates = rnn(self.step, x, initial_states, go_backwards=self.go_backwards, mask=mask, unroll=self.unroll, input_length=input_shape[1])
		self.updates = updates
		if self.stateful:
			for i in range(len(states)):
				self.updates.append((self.states[i], states[i]))
		if self.decode:
			states = states[1:]
		self.state_outputs = states
		if self.return_sequences:
			return outputs
		else:
			return last_output

	def get_initial_states(self, x):
		initial_states = []
		batch_size = self.input_spec[0].shape[0]
		input_length = self.input_spec[0].shape[1]
		if input_length is None:
			input_length = K.shape(x)[1]
		if batch_size is None:
			batch_size = K.shape(x)[0]
		if self.decode:
			input = x
		else:
			input = _get_first_timestep(x)
		for layer in self.model.layers:
			if _isRNN(layer):
				layer_initial_states = []
				for state in layer.states:
					state = self._get_state_from_info(state, input, batch_size, input_length)
					if type(state) != list:
						state = [state]
					layer_initial_states += state
				if not self.state_sync or initial_states == []:
					initial_states += layer_initial_states
				input = layer._step(input, layer_initial_states)[0]
			else:
				input = layer.call(input)
		if self.readout:
			if hasattr(self, 'initial_readout'):
				initial_readout = self._get_state_from_info(self.initial_readout, input, batch_size, input_length)
				initial_states += [initial_readout]
			else:
				initial_states += [K.zeros_like(input)]
		return initial_states

	def reset_states(self):
		batch_size = self.input_spec[0].shape[0]
		input_length = self.input_spec[0].shape[1]
		states = []
		for layer in self.model.layers:
			if _isRNN(layer):
				for state in layer.states:
					assert type(state) in [tuple, list] or 'numpy' in str(type(state)), 'Stateful RNNs require states with static shapes'
					if 'numpy' in str(type(state)):
						states += [K.variable(state)]
					elif type(state) in [list, tuple]:
						state = list(state)
						for i in range(len(state)):
							if state[i] in [-1, 'batch_size']:
								assert type(batch_size) == int, 'Stateful RNNs require states with static shapes'
								state[i] = batch_size
							elif state[i] == 'input_length':
								assert type(input_length) == int, 'Stateful RNNs require states with static shapes'
								state[i] = input_length
						states += [K.variable(np.zeros(state))]
					else:
						states += [state]
				if self.state_sync:
					break
		if self.readout:
			shape = list(self.model.output_shape)
			shape.pop(1)
			states += [K.zeros(shape)]
		self.states = states


	def _get_state_from_info(self, info, input, batch_size, input_length):
		if hasattr(info, '__call__'):
			return info(input)
		elif type(info) in [list, tuple]:
			info = list(info)
			for i in range(len(info)):
				if info[i] in [-1, 'batch_size']:
					info[i] = batch_size
				elif info[i] == 'input_length':
					info[i] = input_length
			if K._BACKEND == 'theano':
				from theano import tensor as k
			else:
				import tensorflow as k
			return k.zeros(info)
		elif 'numpy' in str(type(info)):
			return K.variable(info)
		else:
			return info

	@property
	def trainable_weights(self):
		if not self.model.layers:
			return []
		return self.model.trainable_weights

	@trainable_weights.setter
	def trainable_weights(self, value):
		pass

	@property
	def non_trainable_weights(self):
		if not self.model.layers:
			return []
		return self.model.non_trainable_weights

	@non_trainable_weights.setter
	def non_trainable_weights(self, value):
		pass

	@property
	def weights(self):
		return self.model.weights

	@property
	def regularizers(self):
		if not self.model.layers:
			return []
		return self.model.regularizers

	@regularizers.setter
	def regularizers(self, value):
		pass

	def get_config(self):
		attribs = ['return_sequences', 'go_backwards', 'stateful', 'readout', 'state_sync', 'decode', 'input_length', 'unroll', 'output_length']
		config = {x : getattr(self, x) for x in attribs}
		config['model'] = self.model.get_config()
		base_config = super(RecurrentContainer, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	@classmethod
	def from_config(cls, config):
		model_config = config['model']
		del config['model']
		rc = cls(**config)
		from . import cells
		rc.model = Sequential()
		for layer_config in model_config:
			layer = layer_from_config(layer_config, cells.__dict__)
			rc.add(layer)
		return rc
