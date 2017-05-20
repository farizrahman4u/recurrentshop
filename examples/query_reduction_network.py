import numpy as np
from recurrentshop import RecurrentModel
from keras.models import Model
from keras.layers import Activation, Dense, Embedding, Input, Lambda
from keras.layers import add, concatenate, multiply
from keras import backend as K


# Hyperparameters
batch_size = 128
query_len = 30
story_len = 500
sentence_len = 25
lines_per_story = 20
embedding_dim = 64
vocab_size = 10000


# Get positional encoder matrix
def get_PE_matrix(sentence_len, embedding_dim):
    pe_matrix = np.zeros((embedding_dim, sentence_len), dtype='float32')
    for k in range(embedding_dim):
        for j in range(sentence_len):
            pe_matrix[k][j] = (1 - float(j+1)/float(sentence_len)) - float(k+1)/float(embedding_dim)*(1 - (2*float(j+1))/float(embedding_dim))
    pe_matrix = np.expand_dims(pe_matrix.T, 0)
    return K.constant(pe_matrix)


#############################################################
# Build QRN Cell
#############################################################
def QRN():
    xq = Input(batch_shape=(batch_size, embedding_dim*2))
    # Split into context and query
    xt = Lambda(lambda x, dim: x[:, :dim], arguments={'dim':embedding_dim})(xq)
    qt = Lambda(lambda x, dim: x[:, dim:], arguments={'dim':embedding_dim})(xq)
    
    h_tm1 = Input(batch_shape=(batch_size, embedding_dim))

    zt = Dense(1, activation='sigmoid')(multiply([xt, qt]))
    ch = Dense(embedding_dim, activation='tanh')(concatenate([xt, qt], axis=-1))
    ht = add([multiply([zt, ch]), multiply([Lambda(lambda x: 1-x)(zt), h_tm1])])
    return RecurrentModel(input=xq, output=ht,
                          initial_states=[h_tm1],
                          final_states=[ht],
                          return_sequences=True)

#############################################################
# Build Model
#############################################################
stories = Input(batch_shape=(batch_size, lines_per_story*sentence_len), name='Nemo')
queries = Input(batch_shape=(batch_size, query_len), name='Po')

story_PE_matrix = get_PE_matrix(sentence_len, embedding_dim)
query_PE_matrix = get_PE_matrix(query_len, embedding_dim)

m = Embedding(vocab_size, embedding_dim)(stories)
m = Lambda(lambda x: K.reshape(x, (batch_size* lines_per_story, sentence_len, embedding_dim)))(m)
m = Lambda(lambda x, const: x + K.repeat_elements(const, batch_size*lines_per_story, axis=0), arguments={'const':story_PE_matrix})(m)
m = Lambda(lambda x: K.reshape(x, (batch_size, -1, sentence_len, embedding_dim)))(m)
m = Lambda(lambda x: K.sum(x, axis=1))(m)

q = Embedding(vocab_size, embedding_dim)(queries)
# Add PE encoder matrix
q = Lambda(lambda x, const: x + K.repeat_elements(const, batch_size, axis=0), arguments={'const':query_PE_matrix})(q)
q = Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(q)
q = Lambda(lambda x: K.repeat_elements(x, sentence_len, axis=1))(q)
# Input to RecModel should be a single tensor
mq = concatenate([m, q])

a = QRN()(mq)
a = Lambda(lambda x: x[:, sentence_len - 1, :])(a)
a = Dense(vocab_size)(a)
a = Activation('softmax', name='Steve-Rogers')(a)

model = Model(inputs=[stories, queries], outputs=[a], name='Mufasa')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')