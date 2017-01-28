from recurrentshop import*
from keras.layers import*
from keras.models import*
import numpy as np
import time
import sys


# Script for comparing performance of native keras and recurrentshop stacked RNN implementations
# We observe 20-30% speed ups on GPU


sys.setrecursionlimit(10000000)

# Params

rnn, rnn_cell = LSTM, LSTMCell
depth = 3
input_length = 1000
dim = 10
nb_epoch = 5
unroll = K.backend() == 'tensorflow'

# Random data

x = np.random.random((10, input_length, dim))
y = np.random.random((10, dim))

# Native keras model

model = Sequential()
for i in range(depth):
	model.add(rnn(dim, return_sequences=i != depth-1, input_shape=(input_length, dim), unroll=unroll, consume_less='gpu'))  # We set consume_less = 'gpu' so that both models use the same LSTM implementation.

model.compile(loss='mse', optimizer='sgd')

print('Compiling...')
model.train_on_batch(x[:1], y[:1])  # force compile

start_time = time.time()
model.fit(x, y, nb_epoch=nb_epoch)
end_time = time.time()

keras_time_taken = end_time - start_time

# recurrentshop model

rc = RecurrentContainer(input_length=input_length, unroll=unroll)
for _ in range(depth):
	rc.add(rnn_cell(dim, input_dim=dim))

model = Sequential()
model.add(rc)

model.compile(loss='mse', optimizer='sgd')

print('Compiling...')
model.train_on_batch(x[:1], y[:1])  # force compile

start_time = time.time()
model.fit(x, y, nb_epoch=nb_epoch)
end_time = time.time()

recurrentshop_time_taken = end_time - start_time

speed_up = keras_time_taken / recurrentshop_time_taken


print('Time taken by native keras model: ' + str(int(keras_time_taken)) + ' seconds.')
print('Time taken by recurrentshop model: ' + str(int(recurrentshop_time_taken)) + ' seconds.')
print('Speed up:' + str(speed_up) + 'X')
