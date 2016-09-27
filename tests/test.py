from recurrentshop import*
from keras.models import Sequential
import numpy as np
from keras.layers import*


####
rc = RecurrentContainer()
rc.add(SimpleRNNCell(4, input_dim=5))
rc.add(GRUCell(3))
rc.add(LSTMCell(2))

model = Sequential()
model.add(rc)
model.add(Activation('sigmoid'))
model.compile(loss='mse', optimizer='sgd')

x = np.random.random((100, 10, 5))
y = np.random.random((100, 2))

model.fit(x, y)


####
rc = RecurrentContainer(readout=True)
rc.add(SimpleRNNCell(4, input_dim=5))
rc.add(GRUCell(3))
rc.add(LSTMCell(5))

model = Sequential()
model.add(rc)
model.add(Activation('sigmoid'))
model.compile(loss='mse', optimizer='sgd')

x = np.random.random((100, 10, 5))
y = np.random.random((100, 5))

model.fit(x, y)


####
rc = RecurrentContainer(state_sync=True)
rc.add(GRUCell(4, input_dim=5))
rc.add(GRUCell(4))

model = Sequential()
model.add(rc)
model.add(Activation('sigmoid'))
model.compile(loss='mse', optimizer='sgd')

x = np.random.random((100, 10, 5))
y = np.random.random((100, 4))

model.fit(x, y)
