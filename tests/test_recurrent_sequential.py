from recurrentshop import *
from keras.models import *
from keras.layers import *

x = Input((5,))
h_tm1 = Input((10,))
h = add([Dense(10)(x), Dense(10, use_bias=False)(h_tm1)])
h = Activation('tanh')(h)


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
