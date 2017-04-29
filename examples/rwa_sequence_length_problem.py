'''
Machine Learning on Sequential Data Using a Recurrent Weighted Average
by Jared Ostmeyer and Lindsay Cowell

https://arxiv.org/abs/1703.01253

This is the implementation of 'Classifying by Sequence Length'
experiment mentioned in Section 3.3 of the paper
'''


import numpy as np
from recurrentshop import RecurrentModel
from keras.models import Model
from keras.layers import Dense, Activation, Lambda, Input
from keras.layers import add, concatenate, multiply
from keras import backend as K
from keras import initializers


'''
Training Data

The length of each sequence is randomly drawn from a uniform distribution over
every possible length 0 to T, where T is the maximum possible length of
the sequence. Each step in the sequence is populated with a random number drawn
from a unit normal distribution. Sequences greater than length T /2 are
labeled with 1 while shorter sequences are labeled with 0.
'''


def generate_data(num_samples, max_len):
    data = np.zeros([num_samples, max_len])
    labels = np.zeros([num_samples, 1])

    for sample, label in zip(data, labels):
        length = np.random.randint(0, max_len + 1)
        n = np.random.normal(size=length)
        sample[:length] += n
        if length > max_len / 2:
            label += 1

    data = np.expand_dims(data, axis=-1)
    return data, labels


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

input_dim = 1
output_dim = 250
timesteps = 100
batch_size = 100
n_epochs = 5

####################################################################
# Fetch datasets
####################################################################

train_data, train_labels = generate_data(num_samples=100000, max_len=timesteps)
test_data, test_labels = generate_data(num_samples=100, max_len=timesteps)

####################################################################
# Build and train model
####################################################################

inp = Input((timesteps, input_dim))
out = RWA(input_dim, output_dim)(inp)
out = Dense(1, activation='sigmoid')(out)
model = Model(inp, out)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_data, train_labels, batch_size=batch_size, epochs=n_epochs, validation_data=(test_data, test_labels))
