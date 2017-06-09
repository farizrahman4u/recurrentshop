from recurrentshop import RecurrentModel
from keras.layers import Input, Dense, add, Activation
from keras.models import Model
from keras.utils.test_utils import keras_test
import numpy as np


@keras_test
def test_model():
    x = Input((5,))
    h_tm1 = Input((10,))
    h = add([Dense(10)(x), Dense(10, use_bias=False)(h_tm1)])
    h = Activation('tanh')(h)
    a = Input((7, 5))

    rnn = RecurrentModel(input=x, output=h, initial_states=h_tm1, final_states=h)
    b = rnn(a)
    model = Model(a, b)

    model.compile(loss='mse', optimizer='sgd')
    model.fit(np.random.random((32, 7, 5)), np.random.random((32, 10)))
    model.predict(np.zeros((32, 7, 5)))


@keras_test
def test_state_initializer():
    x = Input((5,))
    h_tm1 = Input((10,))
    h = add([Dense(10)(x), Dense(10, use_bias=False)(h_tm1)])
    h = Activation('tanh')(h)
    a = Input((7, 5))

    rnn = RecurrentModel(input=x, output=h, initial_states=h_tm1, final_states=h, state_initializer='random_normal')
    b = rnn(a)
    model = Model(a, b)

    model.compile(loss='mse', optimizer='sgd')
    model.fit(np.random.random((32, 7, 5)), np.random.random((32, 10)))
    model.predict(np.zeros((32, 7, 5)))


@keras_test
def test_unroll():
    x = Input((5,))
    h_tm1 = Input((10,))
    h = add([Dense(10)(x), Dense(10, use_bias=False)(h_tm1)])
    h = Activation('tanh')(h)
    a = Input((7, 5))

    rnn = RecurrentModel(input=x, output=h, initial_states=h_tm1, final_states=h, unroll=True)
    b = rnn(a)
    model = Model(a, b)

    model.compile(loss='mse', optimizer='sgd')
    model.fit(np.random.random((32, 7, 5)), np.random.random((32, 10)))
    model.predict(np.zeros((32, 7, 5)))


@keras_test
def test_decode():
    x = Input((5,))
    h_tm1 = Input((10,))
    h = add([Dense(10)(x), Dense(10, use_bias=False)(h_tm1)])
    h = Activation('tanh')(h)
    
    a = Input((5,))
    rnn = RecurrentModel(input=x, output=h, initial_states=h_tm1, final_states=h, decode=True, output_length=7)
    b = rnn(a)
    model = Model(a, b)

    model.compile(loss='mse', optimizer='sgd')
    model.fit(np.random.random((32, 5)), np.random.random((32, 7, 10)))
    model.predict(np.zeros((32, 5)))


@keras_test
def test_readout():
    x = Input((5,))
    y_tm1 = Input((5,))
    h_tm1 = Input((5,))
    h = add([Dense(5)(add([x, y_tm1])), Dense(5, use_bias=False)(h_tm1)])
    h = Activation('tanh')(h)

    rnn = RecurrentModel(input=x, initial_states=h_tm1, output=h, final_states=h, readout_input=y_tm1)

    a = Input((7, 5))
    b = rnn(a)
    model = Model(a, b)

    model.compile(loss='mse', optimizer='sgd')
    model.fit(np.random.random((32, 7, 5)), np.random.random((32, 5)))
    model.predict(np.zeros((32, 7, 5)))