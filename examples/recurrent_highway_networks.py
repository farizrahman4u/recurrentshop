'''
Recurrent Highway Networks
-------------------------------------------------------------------------------
Julian Georg Zilly | Rupesh Kumar Srivastava | Jan Koutnik | Jurgen Schmidhuber
https://arxiv.org/abs/1607.03474

This is an implementation of language modeling experiments
on text8 dataset as specified in the paper

Visit https://github.com/julian121266/RecurrentHighwayNetworks for
implementations using Tensorflow, Torch7 and Brainstorm frameworks
and other datasets
'''

from recurrentshop import RecurrentModel
from recurrentshop.advanced_cells import RHNCell
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.layers import add, multiply
from keras.layers import Activation, Embedding
from keras.constraints import max_norm
from keras.initializers import Constant, RandomUniform
from keras.regularizers import l2
from keras.preprocessing.text import Tokenizer
from keras.callbacks import Callback
from keras import backend as K
import numpy as np
import os
import urllib
import zipfile


#
# Hyperparameters
#
batch_size = 128
timesteps = 10
learning_rate = 0.2
hidden_dim = 10
recurrence_depth = 10
weight_decay = 1e-7
lr_decay = 1.04
gradient_clip = 10
embedding_drop = 0.05
output_drop = 0.3
input_drop = 0.3
hidden_drop = 0.05
transform_bias = -4.0
weight_init = RandomUniform(-0.04, 0.04)


def download_data(path):
    print('Downloading data . . .')
    url = "http://mattmahoney.net/dc/text8.zip"
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    urllib.urlretrieve(url, path)
    with zipfile.Zipfile(path) as zf:
        zf.extractall(path=path)


def load_text():
    recurrentshop_directory = os.path.expanduser('~') + '/.recurrentshop'
    datasets_directory = recurrentshop_directory + '/datasets'
    FILE_PATH = os.path.join(recurrentshop_directory, datasets_directory, 'text8')
    if not os.path.exists(FILE_PATH):
        download_data(FILE_PATH)
    raw_text = open(FILE_PATH, 'r').read(100000)

    tokenizer = Tokenizer(filters='', char_level=True, lower=False)
    tokenizer.fit_on_texts(raw_text)
    tokenized_text = tokenizer.texts_to_sequences(raw_text)
    return tokenized_text, len(tokenizer.word_index)


tokenized_text, vocab_size = load_text()
embedding_dim = vocab_size  # Size of character set


def generate_batch(text, batch_size, num_steps):
    raw_data = np.squeeze(np.array(text, dtype=np.int32))
    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    i = 0
    while i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, (i+1)*num_steps]
        if i + 1 >= epoch_size:
            i = 0
        else:
            i += 1
        yield (x, y)


def RHN(input_dim, hidden_dim, depth):
    # Wrapped model
    inp = Input(batch_shape=(batch_size, input_dim))
    state = Input(batch_shape=(batch_size, hidden_dim))
    drop_mask = Input(batch_shape=(batch_size, hidden_dim))
    # To avoid all zero mask causing gradient to vanish
    inverted_drop_mask = Lambda(lambda x: 1.0 - x, output_shape=lambda s: s)(drop_mask)
    drop_mask_2 = Lambda(lambda x: x + 0., output_shape=lambda s: s)(inverted_drop_mask)
    dropped_state = multiply([state, inverted_drop_mask])
    y, new_state = RHNCell(units=hidden_dim, recurrence_depth=depth,
                           kernel_initializer=weight_init,
                           kernel_regularizer=l2(weight_decay),
                           kernel_constraint=max_norm(gradient_clip),
                           bias_initializer=Constant(transform_bias),
                           recurrent_initializer=weight_init,
                           recurrent_regularizer=l2(weight_decay),
                           recurrent_constraint=max_norm(gradient_clip))([inp, dropped_state])
    return RecurrentModel(input=inp, output=y,
                          initial_states=[state, drop_mask],
                          final_states=[new_state, drop_mask_2])


# lr decay Scheduler
class lr_scheduler(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if epoch > 5:
            lr = self.lr / 1.04
            K.set_value(self.model.optimizer.lr, lr)

###########################################
# Build Model
###########################################

inp = Input(batch_shape=(batch_size, timesteps))
x = Dropout(embedding_drop)(inp)
x = Embedding(vocab_size+1, embedding_dim, input_length=timesteps)(inp)
x = Dropout(input_drop)(x)

# Create a dropout mask for variational dropout
drop_mask = Lambda(lambda x: x[:, 0, :1] * 0., output_shape=lambda s: (s[0], 1))(x)

drop_mask = Lambda(lambda x, dim: K.tile(x, (1, dim)),
                   arguments={'dim': hidden_dim},
                   output_shape=(hidden_dim,))(drop_mask)
drop_mask = Lambda(K.ones_like, output_shape=lambda s: s)(drop_mask)
drop_mask = Dropout(hidden_drop)(drop_mask)
zero_init = Lambda(K.zeros_like, output_shape=lambda s:s)(drop_mask)

x = RHN(embedding_dim, hidden_dim, recurrence_depth)(x, initial_state=[zero_init, drop_mask])
x = Dropout(output_drop)(x)
out = Dense(vocab_size+1, activation='softmax')(x)

model = Model(inputs=[inp], outputs=[out])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

data_gen = generate_batch(tokenized_text, batch_size, timesteps)

model.fit_generator(generator=data_gen,
                    steps_per_epoch=(len(tokenized_text)//batch_size),
                    epochs=5,
                    verbose=1,
                    callbacks=[lr_scheduler()])
