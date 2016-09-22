from recurrentshop import*
from keras.models import Sequential
import numpy as np


####
rc = RecurrentContainer()
rc.add(SimpleRNNCell(3, input_dim=4))
rc.add(GRUCell(3))
rc.add(LSTMCell(5))

model = Sequential()
model.add(rc)

model.compile(loss='mse', optimizer='sgd')

x = np.random.random((100, 10, 4))
y = np.random.random((100, 5))

model.fit(x, y)

####
rc = StateSyncRecurrentContainer()
rc.add(LSTMCell(3, input_dim=4))
rc.add(LSTMCell(3))


model = Sequential()
model.add(rc)

model.compile(loss='mse', optimizer='sgd')

x = np.random.random((100, 10, 4))
y = np.random.random((100, 3))

model.fit(x, y)
