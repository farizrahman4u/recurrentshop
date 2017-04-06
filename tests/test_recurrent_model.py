from recurrentshop import RecurrentModel
from keras.models import Model
from keras.layers import *

x = Input((5,))
h_tm1 = Input((10,))
h = add([Dense(10)(x), Dense(10, use_bias=False)(h_tm1)])
h = Activation('tanh')(h)


rnn = RecurrentModel(input=x, output=h, initial_states=h_tm1, final_states=h)
a = Input((7, 5))
b = rnn(a)
model = Model(a, b)

model.compile(loss='mse', optimizer='sgd')
model.fit((np.random.random((32, 7, 5))), np.random.random((32, 10)))
model.predict(np.zeros((32, 7, 5)))


rnn = RecurrentModel(input=x, output=h, initial_states=h_tm1, final_states=h, unroll=True)
b = rnn(a)
model = Model(a, b)

model.compile(loss='mse', optimizer='sgd')
model.fit((np.random.random((32, 7, 5))), np.random.random((32, 10)))
model.predict(np.zeros((32, 7, 5)))

a = Input((5,))
rnn = RecurrentModel(input=x, output=h, initial_states=h_tm1, final_states=h, decode=True, output_length=7)
b = rnn(a)
model = Model(a, b)

model.compile(loss='mse', optimizer='sgd')
model.fit((np.random.random((32, 5))), np.random.random((32, 7, 10)))
model.predict(np.zeros((32, 5)))
