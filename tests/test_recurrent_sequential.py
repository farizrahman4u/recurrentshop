from recurrentshop import *
from keras.models import *
from keras.layers import *


rnn = RecurrentSequential()
rnn.add(LSTMCell(7, input_dim=5))
rnn.add(SimpleRNNCell(8))
rnn.add(GRUCell(10))


a = Input((7, 5))
b = rnn(a)

model = Model(a, b)

model.compile(loss='mse', optimizer='sgd')
model.fit((np.random.random((32, 7, 5))), np.random.random((32, 10)))
model.predict(np.random.random((32, 7, 5)))


rnn = RecurrentSequential(unroll=True)
rnn.add(LSTMCell(7, input_dim=5))
rnn.add(SimpleRNNCell(8))
rnn.add(GRUCell(10))


b = rnn(a)

model = Model(a, b)

model.compile(loss='mse', optimizer='sgd')
model.fit((np.random.random((32, 7, 5))), np.random.random((32, 10)))
model.predict(np.random.random((32, 7, 5)))


rnn = RecurrentSequential(state_sync=True)
rnn.add(LSTMCell(10, input_dim=5))
rnn.add(LSTMCell(10))
rnn.add(LSTMCell(10))


b = rnn(a)

model = Model(a, b)

model.compile(loss='mse', optimizer='sgd')
model.fit((np.random.random((32, 7, 5))), np.random.random((32, 10)))
model.predict(np.random.random((32, 7, 5)))



rnn = RecurrentSequential(state_sync=True, unroll=True)
rnn.add(LSTMCell(10, input_dim=5))
rnn.add(LSTMCell(10))
rnn.add(LSTMCell(10))


b = rnn(a)

model = Model(a, b)

model.compile(loss='mse', optimizer='sgd')
model.fit((np.random.random((32, 7, 5))), np.random.random((32, 10)))
model.predict(np.random.random((32, 7, 5)))


# Decoders

a = Input((5,))

rnn = RecurrentSequential(decode=True, output_length=7)
rnn.add(LSTMCell(10, input_dim=5))
rnn.add(LSTMCell(10))
rnn.add(LSTMCell(10))

b = rnn(a)

model = Model(a, b)

model.compile(loss='mse', optimizer='sgd')
model.fit((np.random.random((32, 5))), np.random.random((32, 7, 10)))
model.predict(np.random.random((32, 5)))


rnn = RecurrentSequential(state_sync=True, decode=True, output_length=7)
rnn.add(LSTMCell(10, input_dim=5))
rnn.add(LSTMCell(10))
rnn.add(LSTMCell(10))

b = rnn(a)

model = Model(a, b)

model.compile(loss='mse', optimizer='sgd')
model.fit((np.random.random((32, 5))), np.random.random((32, 7, 10)))
model.predict(np.random.random((32, 5)))

rnn = RecurrentSequential(decode=True, output_length=7, unroll=True)
rnn.add(LSTMCell(10, input_dim=5))
rnn.add(LSTMCell(10))
rnn.add(LSTMCell(10))

b = rnn(a)

model = Model(a, b)

model.compile(loss='mse', optimizer='sgd')
model.fit((np.random.random((32, 5))), np.random.random((32, 7, 10)))
model.predict(np.random.random((32, 5)))

rnn = RecurrentSequential(state_sync=True, decode=True, output_length=7, unroll=True)
rnn.add(LSTMCell(10, input_dim=5))
rnn.add(LSTMCell(10))
rnn.add(LSTMCell(10))

b = rnn(a)

model = Model(a, b)

model.compile(loss='mse', optimizer='sgd')
model.fit((np.random.random((32, 5))), np.random.random((32, 7, 10)))
model.predict(np.random.random((32, 5)))


# Readout

a = Input((7, 5))

rnn = RecurrentSequential(readout=True)
rnn.add(LSTMCell(7, input_dim=5))
rnn.add(SimpleRNNCell(8))
rnn.add(GRUCell(5))

b = rnn(a)

model = Model(a, b)

model.compile(loss='mse', optimizer='sgd')
model.fit((np.random.random((32, 7, 5))), np.random.random((32, 5)))
model.predict(np.random.random((32, 7, 5)))

rnn = RecurrentSequential(readout=True, unroll=True)
rnn.add(LSTMCell(7, input_dim=5))
rnn.add(SimpleRNNCell(8))
rnn.add(GRUCell(5))

b = rnn(a)

model = Model(a, b)

model.compile(loss='mse', optimizer='sgd')
model.fit((np.random.random((32, 7, 5))), np.random.random((32, 5)))
model.predict(np.random.random((32, 7, 5)))

rnn = RecurrentSequential(readout=True, state_sync=True)
rnn.add(LSTMCell(5, input_dim=5))
rnn.add(LSTMCell(5))
rnn.add(LSTMCell(5))

b = rnn(a)

model = Model(a, b)

model.compile(loss='mse', optimizer='sgd')
model.fit((np.random.random((32, 7, 5))), np.random.random((32, 5)))
model.predict(np.random.random((32, 7, 5)))


rnn = RecurrentSequential(readout=True, state_sync=True, unroll=True)
rnn.add(LSTMCell(5, input_dim=5))
rnn.add(LSTMCell(5))
rnn.add(LSTMCell(5))

b = rnn(a)

model = Model(a, b)

model.compile(loss='mse', optimizer='sgd')
model.fit((np.random.random((32, 7, 5))), np.random.random((32, 5)))
model.predict(np.random.random((32, 7, 5)))



# Decoder + readout

a = Input((5,))

rnn = RecurrentSequential(decode=True, output_length=7, readout=True)
rnn.add(LSTMCell(5, input_dim=5))
rnn.add(LSTMCell(5))
rnn.add(LSTMCell(5))

b = rnn(a)

model = Model(a, b)

model.compile(loss='mse', optimizer='sgd')
model.fit((np.random.random((32, 5))), np.random.random((32, 7, 5)))
model.predict(np.random.random((32, 5)))

# teacher forcing

a = Input((7, 5))

rnn = RecurrentSequential(readout=True, teacher_force=True)
rnn.add(LSTMCell(7, input_dim=5))
rnn.add(SimpleRNNCell(8))
rnn.add(GRUCell(5))

ground_truth = Input((7, 5))

b = rnn(a, ground_truth=ground_truth)

model = Model([a, ground_truth], b)

model.compile(loss='mse', optimizer='sgd')
model.fit([np.random.random((32, 7, 5)), np.random.random((32, 7, 5))], np.random.random((32, 5)))
model.predict([np.random.random((32, 7, 5))] * 2)
