from recurrentshop import *
from keras.models import Sequential
import numpy as np
from keras.layers import *
from keras import backend as K
from keras.utils.test_utils import keras_test

class TestDecoder(RNNCell):

	def build(self, input_shape):
		def step(x, states):
			return x[:, 0, :], states
		self.step = step
		self.states = []
		super(TestDecoder, self).build(input_shape)

	def get_output_shape_for(self, input_shape):
		return (input_shape[0], input_shape[2])


@keras_test
def test_all():
	rc = RecurrentContainer()
	rc.add(SimpleRNNCell(4, input_dim=5))
	rc.add(GRUCell(3))
	rc.add(LSTMCell(2))

	model = Sequential()
	model.add(rc)
	model.compile(loss='mse', optimizer='sgd')

	x = np.random.random((100, 10, 5))
	y = np.random.random((100, 2))

	model.fit(x, y, nb_epoch=1)


	#### with readout=True
	rc = RecurrentContainer(readout=True)
	rc.add(SimpleRNNCell(4, input_dim=5))
	rc.add(GRUCell(3))
	rc.add(LSTMCell(5))

	model = Sequential()
	model.add(rc)
	model.compile(loss='mse', optimizer='sgd')

	x = np.random.random((100, 10, 5))
	y = np.random.random((100, 5))

	model.fit(x, y, nb_epoch=1)


	#### with state_sync=True
	rc = RecurrentContainer(state_sync=True)
	rc.add(GRUCell(3, input_dim=4))
	rc.add(GRUCell(3))

	model = Sequential()
	model.add(rc)
	model.compile(loss='mse', optimizer='sgd')

	x = np.random.random((100, 10, 4))
	y = np.random.random((100, 3))

	model.fit(x, y, nb_epoch=1)

	#### with state_sync=True, readout=True
	rc = RecurrentContainer(state_sync=True, readout=True)
	rc.add(GRUCell(4, input_dim=4))
	rc.add(GRUCell(4))

	model = Sequential()
	model.add(rc)
	model.compile(loss='mse', optimizer='sgd')

	x = np.random.random((100, 10, 4))
	y = np.random.random((100, 4))

	model.fit(x, y, nb_epoch=1)

	#### with dropout

	rc = RecurrentContainer()
	rc.add(SimpleRNNCell(3, input_dim=3))
	rc.add(Dropout(0.5))
	model = Sequential()
	model.add(rc)
	model.compile(loss='mse', optimizer='sgd')

	x = np.random.random((100, 3, 3))
	y = np.random.random((100, 3))

	model.fit(x, y, nb_epoch=1)

@keras_test
def test_RecurrentContainer():
	rc = RecurrentContainer(output_length=5, decode=True)
	rc.add(TestDecoder(input_shape=(10, 10)))

	model = Sequential()
	model.add(rc)
	model.compile(loss='mse', optimizer='sgd')

	x = np.zeros((100, 10, 10))
	y = np.zeros((100, 5, 10))
	model.fit(x, y, nb_epoch=1)
