from recurrentshop import RecurrentSequential
from recurrentshop.cells import *
from recurrentshop.advanced_cells import *
from keras.models import Model
from keras.layers import Input
from keras.utils.test_utils import keras_test
import numpy as np


@keras_test
def test_sequential():
    rnn = RecurrentSequential()
    rnn.add(LSTMCell(output_dim=7, input_dim=5))
    rnn.add(SimpleRNNCell(8))
    rnn.add(GRUCell(10))

    a = Input((7, 5))
    b = rnn(a)

    model = Model(a, b)

    model.compile(loss='mse', optimizer='sgd')
    model.fit((np.random.random((12, 7, 5))), np.random.random((12, 10)))
    model.predict(np.random.random((12, 7, 5)))


@keras_test
def test_state_initializer():
    rnn = RecurrentSequential(state_initializer='random_normal')
    rnn.add(LSTMCell(7, input_dim=5))
    rnn.add(SimpleRNNCell(8))
    rnn.add(GRUCell(10))

    a = Input((7, 5))
    b = rnn(a)

    model = Model(a, b)

    model.compile(loss='mse', optimizer='sgd')
    model.fit((np.random.random((12, 7, 5))), np.random.random((12, 10)))
    model.predict(np.random.random((12, 7, 5)))


@keras_test
def test_state_initializer_as_list():
    rnn = RecurrentSequential(state_initializer=['random_normal', 'glorot_uniform'])
    rnn.add(LSTMCell(7, batch_input_shape=(12, 5)))
    rnn.add(SimpleRNNCell(8))
    rnn.add(GRUCell(10))

    a = Input((7, 5))
    b = rnn(a)

    model = Model(a, b)

    model.compile(loss='mse', optimizer='sgd')
    model.fit((np.random.random((12, 7, 5))), np.random.random((12, 10)))
    model.predict(np.random.random((12, 7, 5)))


@keras_test
def test_unroll():
    rnn = RecurrentSequential(unroll=True)
    rnn.add(LSTMCell(7, input_dim=5))
    rnn.add(SimpleRNNCell(8))
    rnn.add(GRUCell(10))

    a = Input((7, 5))
    b = rnn(a)

    model = Model(a, b)

    model.compile(loss='mse', optimizer='sgd')
    model.fit((np.random.random((12, 7, 5))), np.random.random((12, 10)))
    model.predict(np.random.random((12, 7, 5)))


@keras_test
def test_state_sync():
    rnn = RecurrentSequential(state_sync=True)
    rnn.add(LSTMCell(10, input_dim=5))
    rnn.add(LSTMCell(10))
    rnn.add(LSTMCell(10))

    a = Input((7, 5))
    b = rnn(a)

    model = Model(a, b)

    model.compile(loss='mse', optimizer='sgd')
    model.fit((np.random.random((12, 7, 5))), np.random.random((12, 10)))
    model.predict(np.random.random((12, 7, 5)))


@keras_test
def test_state_sync_unroll():
    rnn = RecurrentSequential(state_sync=True, unroll=True)
    rnn.add(LSTMCell(10, input_dim=5))
    rnn.add(LSTMCell(10))
    rnn.add(LSTMCell(10))

    a = Input((7, 5))
    b = rnn(a)

    model = Model(a, b)

    model.compile(loss='mse', optimizer='sgd')
    model.fit((np.random.random((12, 7, 5))), np.random.random((12, 10)))
    model.predict(np.random.random((12, 7, 5)))


# Decoders
@keras_test
def test_decode():
    a = Input((5,))

    rnn = RecurrentSequential(decode=True, output_length=7)
    rnn.add(LSTMCell(10, input_dim=5))
    rnn.add(LSTMCell(10))
    rnn.add(LSTMCell(10))

    b = rnn(a)

    model = Model(a, b)

    model.compile(loss='mse', optimizer='sgd')
    model.fit((np.random.random((12, 5))), np.random.random((12, 7, 10)))
    model.predict(np.random.random((12, 5)))


@keras_test
def test_readout_state_sync():
    a = Input((5,))
    rnn = RecurrentSequential(state_sync=True, decode=True, output_length=7)
    rnn.add(LSTMCell(10, input_dim=5))
    rnn.add(LSTMCell(10))
    rnn.add(LSTMCell(10))

    b = rnn(a)

    model = Model(a, b)

    model.compile(loss='mse', optimizer='sgd')
    model.fit((np.random.random((12, 5))), np.random.random((12, 7, 10)))
    model.predict(np.random.random((12, 5)))


@keras_test
def test_decode_unroll():
    a = Input((5,))
    rnn = RecurrentSequential(decode=True, output_length=7, unroll=True)
    rnn.add(LSTMCell(10, input_dim=5))
    rnn.add(LSTMCell(10))
    rnn.add(LSTMCell(10))

    b = rnn(a)

    model = Model(a, b)

    model.compile(loss='mse', optimizer='sgd')
    model.fit((np.random.random((12, 5))), np.random.random((12, 7, 10)))
    model.predict(np.random.random((12, 5)))


@keras_test
def test_decode_unroll_state_sync():
    a = Input((5,))
    rnn = RecurrentSequential(state_sync=True, decode=True, output_length=7, unroll=True)
    rnn.add(LSTMCell(10, input_dim=5))
    rnn.add(LSTMCell(10))
    rnn.add(LSTMCell(10))

    b = rnn(a)

    model = Model(a, b)

    model.compile(loss='mse', optimizer='sgd')
    model.fit((np.random.random((12, 5))), np.random.random((12, 7, 10)))
    model.predict(np.random.random((12, 5)))


# Readout
@keras_test
def test_readout():
    a = Input((7, 5))

    rnn = RecurrentSequential(readout=True)
    rnn.add(LSTMCell(7, input_dim=5))
    rnn.add(SimpleRNNCell(8))
    rnn.add(GRUCell(5))

    b = rnn(a)

    model = Model(a, b)

    model.compile(loss='mse', optimizer='sgd')
    model.fit((np.random.random((12, 7, 5))), np.random.random((12, 5)))
    model.predict(np.random.random((12, 7, 5)))


@keras_test
def test_readout_unroll():
    a = Input((7, 5))
    rnn = RecurrentSequential(readout=True, unroll=True)
    rnn.add(LSTMCell(7, input_dim=5))
    rnn.add(SimpleRNNCell(8))
    rnn.add(GRUCell(5))

    b = rnn(a)

    model = Model(a, b)

    model.compile(loss='mse', optimizer='sgd')
    model.fit((np.random.random((12, 7, 5))), np.random.random((12, 5)))
    model.predict(np.random.random((12, 7, 5)))


@keras_test
def test_readout_state_sync():
    a = Input((7, 5))
    rnn = RecurrentSequential(readout=True, state_sync=True)
    rnn.add(LSTMCell(5, input_dim=5))
    rnn.add(LSTMCell(5))
    rnn.add(LSTMCell(5))

    b = rnn(a)

    model = Model(a, b)

    model.compile(loss='mse', optimizer='sgd')
    model.fit((np.random.random((12, 7, 5))), np.random.random((12, 5)))
    model.predict(np.random.random((12, 7, 5)))


@keras_test
def test_readout_state_sync_unroll():
    a = Input((7, 5))
    rnn = RecurrentSequential(readout=True, state_sync=True, unroll=True)
    rnn.add(LSTMCell(5, input_dim=5))
    rnn.add(LSTMCell(5))
    rnn.add(LSTMCell(5))

    b = rnn(a)

    model = Model(a, b)

    model.compile(loss='mse', optimizer='sgd')
    model.fit((np.random.random((12, 7, 5))), np.random.random((12, 5)))
    model.predict(np.random.random((12, 7, 5)))


# Decoder + readout
@keras_test
def test_decoder_readout():
    a = Input((5,))

    rnn = RecurrentSequential(decode=True, output_length=7, readout=True)
    rnn.add(LSTMCell(5, input_dim=5))
    rnn.add(LSTMCell(5))
    rnn.add(LSTMCell(5))

    b = rnn(a)

    model = Model(a, b)

    model.compile(loss='mse', optimizer='sgd')
    model.fit((np.random.random((12, 5))), np.random.random((12, 7, 5)))
    model.predict(np.random.random((12, 5)))

# teacher forcing


@keras_test
def test_teacher_force():
    a = Input((7, 5))

    rnn = RecurrentSequential(readout=True, teacher_force=True)
    rnn.add(LSTMCell(7, input_dim=5))
    rnn.add(SimpleRNNCell(8))
    rnn.add(GRUCell(5))

    ground_truth = Input((7, 5))

    b = rnn(a, ground_truth=ground_truth)

    model = Model([a, ground_truth], b)

    model.compile(loss='mse', optimizer='sgd')
    model.fit([np.random.random((12, 7, 5)), np.random.random((12, 7, 5))], np.random.random((12, 5)))
    model.predict([np.random.random((12, 7, 5))] * 2)


@keras_test
def test_serialisation():
    rnn = RecurrentSequential()
    rnn.add(LSTMCell(output_dim=7, input_dim=5))
    rnn.add(SimpleRNNCell(8))
    rnn.add(GRUCell(10))

    rnn_config = rnn.get_config()
    recovered_rnn = RecurrentSequential.from_config(rnn_config)

    a = Input((7, 5))
    b = recovered_rnn(a)

    model = Model(a, b)

    model.compile(loss='mse', optimizer='sgd')
    model.fit((np.random.random((12, 7, 5))), np.random.random((12, 10)))
    model.predict(np.random.random((12, 7, 5)))


@keras_test
def test_advanced_cells():
    rnn = RecurrentSequential()
    rnn.add(RHNCell(10, recurrence_depth=2, input_dim=5))

    a = Input((7, 5))
    b = rnn(a)

    model = Model(a, b)

    model.compile(loss='mse', optimizer='sgd')
    model.fit((np.random.random((12, 7, 5))), np.random.random((12, 10)))
    model.predict(np.random.random((12, 7, 5)))