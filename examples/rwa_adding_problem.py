'''
Machine Learning on Sequential Data Using a Recurrent Weighted Average
by Jared Ostmeyer and Lindsay Cowell

https://arxiv.org/abs/1703.01253

This is the implementation of 'Adding Problem'
mentioned in Section 3.5 of the paper
'''


import numpy as np
from recurrentshop import RecurrentModel
from keras.models import Model
from keras.layers import Dense, Activation, Lambda, Input
from keras.layers import add, concatenate, multiply
from keras import backend as K
from keras import initializers


'''
Training data

The input sequence consists of two dimensions at each step. The first dimension 
serves as an indicator marking the value to add while the second dimension is the
actual number to be added and is drawn at random from a uniform
distribution over [0, 1]. The target value is the sum of the two numbers
that has `1` in the first dimernsion. Only two steps in the entire
sequence will have an indicator of 1, leaving the indicator 0 everywhere else.
'''


def generate_data(num_samples, max_len):
    values = np.random.normal(size=[num_samples, max_len, 1])
    mask = np.zeros([num_samples, max_len, 1])
    answers = np.zeros([num_samples, 1])

    for i in range(num_samples):
        j1, j2 = 0, 0
        while j1 == j2:
            j1 = np.random.randint(max_len)
            j2 = np.random.randint(max_len)
        mask[i, (j1, j2)] = 1.0
        answers[i] = np.sum(values[i]*mask[i])
    data = np.concatenate((values, mask), 2)
    return data, answers


#####################################################################
# RWA layer
#####################################################################

def RWA(input_dim, output_dim):
    x = Input((input_dim, ))
    h_tm1 = Input((output_dim, ))
    n_tm1 = Input((output_dim, ))
    d_tm1 = Input((output_dim, ))

    x_h = concatenate([x, h_tm1])

    u = Dense(output_dim)(x)
    g = Dense(output_dim, activation='tanh')(x_h)

    a = Dense(output_dim, use_bias=False)(x_h)
    e_a = Lambda(lambda x: K.exp(x))(a)

    z = multiply([u, g])
    nt = add([n_tm1, multiply([z, e_a])])
    dt = add([d_tm1, e_a])
    dt = Lambda(lambda x: 1.0 / x)(dt)
    ht = multiply([nt, dt])
    ht = Activation('tanh')(ht)

    return RecurrentModel(input=x, output=ht,
                          initial_states=[h_tm1, n_tm1, d_tm1],
                          final_states=[ht, nt, dt],
                          state_initializer=[initializers.random_normal(stddev=1.0)])


#####################################################################
# Settings
#####################################################################

input_dim = 2
output_dim = 250
timesteps = 100
batch_size = 100
n_epochs = 10

####################################################################
# Fetch datasets
####################################################################
print('Generating train data')
train_data, train_labels = generate_data(num_samples=100000, max_len=timesteps)
print('Generating test data')
test_data, test_labels = generate_data(num_samples=10000, max_len=timesteps)

####################################################################
# Build and train model
####################################################################

inp = Input((timesteps, input_dim))
out = RWA(input_dim, output_dim)(inp)
out = Dense(1)(out)
model = Model(inp, out)

model.compile(loss='mse', optimizer='adam')
model.fit(train_data, train_labels, batch_size=batch_size, epochs=n_epochs, validation_data=(test_data, test_labels))
