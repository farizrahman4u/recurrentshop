import recurrentshop
from keras.models import Sequential, Model, model_from_json
from keras.engine import Input
import numpy as np
from keras.layers import *
from keras import backend as K
from keras.utils.test_utils import keras_test

def cell_test(layer_cls, kwargs={}, input_shape=None, input_dtype=None,
			   input_data=None, expected_output=None,
			   expected_output_dtype=None, fixed_batch_size=False):
	if input_data is None:
		assert input_shape
		if not input_dtype:
			input_dtype = K.floatx()
		input_data = (10 * np.random.random(input_shape)).astype(input_dtype)
	elif input_shape is None:
		input_shape = input_data.shape

	if expected_output_dtype is None:
		expected_output_dtype = input_dtype

	# instantiation
	layer = layer_cls(**kwargs).get_layer()

	# test get_weights , set_weights
	weights = layer.get_weights()
	layer.set_weights(weights)

	# test and instantiation from weights
	if 'weights' in inspect.getargspec(layer_cls.__init__):
		kwargs['weights'] = weights
		layer = layer_cls(**kwargs).get_layer()

	# test in functional API
	if fixed_batch_size:
		x = Input(batch_shape=input_shape, dtype=input_dtype)
	else:
		x = Input(shape=input_shape[1:], dtype=input_dtype)
	y = layer(x)
	assert K.dtype(y) == expected_output_dtype

	model = Model(input=x, output=y)
	model.compile('rmsprop', 'mse')

	expected_output_shape = layer.get_output_shape_for(input_shape)
	actual_output = model.predict(input_data)
	actual_output_shape = actual_output.shape
	assert expected_output_shape == actual_output_shape
	if expected_output is not None:
		assert_allclose(actual_output, expected_output, rtol=1e-3)

	# test serialization
	model_config = model.get_config()
	model = Model.from_config(model_config, custom_objects=recurrentshop.__dict__)
	model.compile('rmsprop', 'mse')

	# test as first layer in Sequential API
	layer_config = layer.get_config()
	layer_config['batch_input_shape'] = input_shape
	layer = layer.__class__.from_config(layer_config)

	model = Sequential()
	model.add(layer)
	model.compile('rmsprop', 'mse')
	actual_output = model.predict(input_data)
	actual_output_shape = actual_output.shape
	assert expected_output_shape == actual_output_shape
	if expected_output is not None:
		assert_allclose(actual_output, expected_output, rtol=1e-3)

	# test JSON serialization
	json_model = model.to_json()
	model = model_from_json(json_model)

	# for further checks in the caller function
	return actual_output

@keras_test
def test_SimpleRNNCell():
	cell_test(recurrentshop.SimpleRNNCell, kwargs={'output_dim':4, 'input_dim': 5}, input_shape=(10, 10, 5), fixed_batch_size=True)

@keras_test
def test_GRUCell():
	cell_test(recurrentshop.GRUCell, kwargs={'output_dim':4, 'input_dim': 5}, input_shape=(10, 10, 5), fixed_batch_size=True)

@keras_test
def test_LSTMCell():
	cell_test(recurrentshop.LSTMCell, kwargs={'output_dim':4, 'input_dim': 5}, input_shape=(10, 10, 5), fixed_batch_size=True)



if __name__ == '__main__':
	test_SimpleRNNCell()
	test_GRUCell()
	test_LSTMCell()
