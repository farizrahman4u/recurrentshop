'''
Query Reduction Networks for Question Answering
Minjoon Seo | Sewon Min | Ali Farhadi | Hannaneh Hajishirzi

https://arxiv.org/pdf/1606.04582.pdf

Experiment run on BaBI task 1
'''


import numpy as np
from recurrentshop import RecurrentModel
from keras.models import Model
from keras.layers import Activation, Dense, Embedding, Input, Lambda
from keras.layers import add, concatenate, multiply
from keras.layers.wrappers import Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import Constant
from keras import backend as K
import tarfile
import urllib
import os

# Hyperparameters
batch_size = 20
query_len = 4
sentence_len = 7
lines_per_story = 2
embedding_dim = 50
vocab_size = 34

tokenizer = Tokenizer()


def _download_data(path):
    url = "http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz"
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, 'bAbI-Tasks-1-20.tar.gz')
    urllib.urlretrieve(url, file_path)
    with tarfile.open(file_path) as tarf:
        tarf.extractall()


def load_data():
    recurrentshop_directory = os.path.expanduser('~') + '/.recurrentshop'
    datasets_directory = recurrentshop_directory + '/datasets'
    babi_path = os.path.join(recurrentshop_directory, datasets_directory, 'tasks_1-20_v1-2')
    if not os.path.exists(babi_path):
        _download_data(babi_path)

    train_path = os.path.join(babi_path, 'en', 'qa1_single-supporting-fact_train.txt')
    test_path = os.path.join(babi_path, 'en', 'qa1_single-supporting-fact_test.txt')

    def fetch_file(path):
        with open(path, 'r') as f:
            text = f.readlines()
        tokenizer.fit_on_texts(text)
        text = tokenizer.texts_to_sequences(text)
        stories = []
        queries = []
        answers = []
        for i in range(0, len(text), 3):
            story = np.append(pad_sequences([text[i][1:]], maxlen=sentence_len)[0],
                              pad_sequences([text[i + 1][1:]], maxlen=sentence_len)[0])
            stories.append(story)
            queries.append(text[i + 2][:-2])
            answers.append(text[i + 2][-2])
        return (np.asarray(stories), np.asarray(queries), np.asarray(answers))

    train_data = fetch_file(train_path)
    test_data = fetch_file(test_path)

    return train_data, test_data


# Get positional encoder matrix
def get_PE_matrix(sentence_len, embedding_dim):
    pe_matrix = np.zeros((embedding_dim, sentence_len), dtype='float32')
    for k in range(embedding_dim):
        for j in range(sentence_len):
            pe_matrix[k][j] = (1 - float(j + 1) / float(sentence_len)) - float(k + 1) / float(embedding_dim) * (1 - (2 * float(j + 1)) / float(embedding_dim))
    pe_matrix = np.expand_dims(pe_matrix.T, 0)
    return pe_matrix


#
# Build QRN Cell
#
def QRNcell():
    xq = Input(batch_shape=(batch_size, embedding_dim * 2))
    # Split into context and query
    xt = Lambda(lambda x, dim: x[:, :dim], arguments={'dim': embedding_dim},
                output_shape=lambda s: (s[0], s[1] / 2))(xq)
    qt = Lambda(lambda x, dim: x[:, dim:], arguments={'dim': embedding_dim},
                output_shape=lambda s: (s[0], s[1] / 2))(xq)

    h_tm1 = Input(batch_shape=(batch_size, embedding_dim))

    zt = Dense(1, activation='sigmoid', bias_initializer=Constant(2.5))(multiply([xt, qt]))
    zt = Lambda(lambda x, dim: K.repeat_elements(x, dim, axis=1), arguments={'dim': embedding_dim})(zt)
    ch = Dense(embedding_dim, activation='tanh')(concatenate([xt, qt], axis=-1))
    rt = Dense(1, activation='sigmoid')(multiply([xt, qt]))
    rt = Lambda(lambda x, dim: K.repeat_elements(x, dim, axis=1), arguments={'dim': embedding_dim})(rt)
    ht = add([multiply([zt, ch, rt]), multiply([Lambda(lambda x: 1 - x, output_shape=lambda s: s)(zt), h_tm1])])
    return RecurrentModel(input=xq, output=ht, initial_states=[h_tm1], final_states=[ht], return_sequences=True)


#
# Load data
#

train_data, test_data = load_data()

train_stories, train_queries, train_answers = train_data
valid_stories, valid_queries, valid_answers = test_data

#
# Build Model
#

stories = Input(batch_shape=(batch_size, lines_per_story * sentence_len))
queries = Input(batch_shape=(batch_size, query_len))

story_PE_matrix = get_PE_matrix(sentence_len, embedding_dim)
query_PE_matrix = get_PE_matrix(query_len, embedding_dim)
QRN = Bidirectional(QRNcell(), merge_mode='sum')
embedding = Embedding(vocab_size + 1, embedding_dim)
m = embedding(stories)
m = Lambda(lambda x: K.reshape(x, (batch_size * lines_per_story, sentence_len, embedding_dim)),
           output_shape=lambda s: (batch_size * lines_per_story, sentence_len, embedding_dim))(m)
# Add PE encoder matrix
m = Lambda(lambda x, const: x + np.tile(const, (batch_size * lines_per_story, 1, 1)), arguments={'const': story_PE_matrix},
           output_shape=lambda s: s)(m)
m = Lambda(lambda x: K.reshape(x, (batch_size, lines_per_story, sentence_len, embedding_dim)),
           output_shape=lambda s: (batch_size, lines_per_story, sentence_len, embedding_dim))(m)
m = Lambda(lambda x: K.sum(x, axis=2),
           output_shape=lambda s: (s[0], s[1], s[3]))(m)

q = embedding(queries)
# Add PE encoder matrix
q = Lambda(lambda x, const: x + np.tile(const, (batch_size, 1, 1)), arguments={'const': query_PE_matrix},
           output_shape=lambda s: s)(q)
q = Lambda(lambda x: K.sum(x, axis=1, keepdims=True),
           output_shape=lambda s: (s[0], 1, s[2]))(q)
q = Lambda(lambda x: K.tile(x, (1, lines_per_story, 1)),
           output_shape=lambda s: (s[0], lines_per_story, s[2]))(q)
# Input to RecModel should be a single tensor
mq = concatenate([m, q])
# Call the RecurrentModel
a = QRN(mq)
mq = concatenate([m, a])
a = QRN(mq)
a = Lambda(lambda x: x[:, lines_per_story - 1, :],
           output_shape=lambda s: (s[0], s[2]))(a)
a = Dense(vocab_size)(a)
a = Activation('softmax')(a)

model = Model(inputs=[stories, queries], outputs=[a])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit([train_stories, train_queries], train_answers,
          batch_size=batch_size,
          verbose=2,
          epochs=100,
          validation_data=([valid_stories, valid_queries], valid_answers))
