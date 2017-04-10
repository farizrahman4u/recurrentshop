# Recurrent Shop
Framework for building complex recurrent neural networks with Keras


Ability to easily iterate over different neural network architectures is key to doing machine learning research. While deep learning libraries like [Keras](https://www.keras.io) makes it very easy to prototype new layers and models, writing custom recurrent neural networks is harder than it needs to be in almost all popular deep learning libraries available today. One key missing feature in these libraries is reusable RNN cells. Most libraries provide layers (such as LSTM, GRU etc), which can only be used as is, and not be easily embedded in a bigger RNN. Writing the RNN logic itself can be tiresome at times. For example in Keras, information about the states (shape and initial value) are provided by writing two seperate functions, `get_initial_states` and `reset_states` (for stateful version). There are many architectures whose implementation is not trivial using modern deep learning libraries, such as:

* Synchronising the states of all the layers in a RNN stack.
* Feeding back the output of the last layer of a RNN stack to the first layer in next time step (readout).
* Decoders : RNNs who can look at the whole of the input sequence / vector at every time step.
* Teacher forcing : Using the ground truth at time t-1 for predicting at time t during training.
* Nested RNNs

Recurrent shop adresses these issues by letting the user write RNNs of arbitrary complexity using Keras's functional API. In other words, the user builds a standard Keras model which defines the logic of the RNN for a single timestep, and RecurrentShop converts this model into a `Recurrent` instance, which is capable of processing sequences.



------------------

## Writing a Simple RNN
 
```python
# The RNN logic is written using Keras's functional API.
# Which means we use Keras layers instead of theano/tensorflow ops
from keras.layers import *
from keras.models import *
from recurrentshop import *


x_t = Input(5,)) # The input to the RNN at time t
h_tm1 = Input((10,))  # Previous hidden state

# Compute new hidden state
h_t = add([Dense(10)(x_t), Dense(10, use_bias=False)(h_tm1)])

# tanh activation
h_t = Activation('tanh')(h_t)

rnn = RecurrentModel(input=x_t, initial_states=[h_tm1], output=h_t, output_states=[h_t])

# rnn is a standard Keras `Recurrent` instance. It accepts arguments such as unroll, return_sequences etc

# Run the RNN over a random sequence

x = Input((7,5))
y = rnn(x)

model = Model(x, y)

model.predict(np.random((7, 5)))

```



## Recurrent container

```python

rc = RecurrentContainer()
rc.add(SimpleRNNCell(10, input_dim=20))
rc.add(Activation('tanh'))
```

## Stacking RNN cells

```python

rc = RecurrentContainer()
rc.add(SimpleRNNCell(10, input_dim=20))
rc.add(SimpleRNNCell(10))
rc.add(SimpleRNNCell(10))
rc.add(SimpleRNNCell(10))
rc.add(Activation('tanh'))

```

## State synchronization

```python
# All cells will use the same state(s)

rc = RecurrentContainer(state_sync=True)
rc.add(SimpleRNNCell(10, input_dim=20))
rc.add(SimpleRNNCell(10))
rc.add(SimpleRNNCell(10))
rc.add(SimpleRNNCell(10))
rc.add(Activation('tanh'))
```

## Readout

```python
# Output of the final layer in the previous time step is available to the first layer(added to the input by default)

rc = RecurrentContainer(readout=True)
rc.add(SimpleRNNCell(10, input_dim=20))
rc.add(SimpleRNNCell(10))
rc.add(SimpleRNNCell(10))
rc.add(SimpleRNNCell(10))
rc.add(Activation('tanh'))
```

## Decoder

```python
# Here we decode a vector into a sequence of vectors. The input could also be a sequence, such as in the case of Attention models, where the whole input sequence is available to the RNN at every time step

# In this case, input to rc is a 2d vector, not a sequence

rc = RecurrentContainer(decode=True, output_length=10)
rc.add(SimpleRNNCell(10, input_dim=20))
```

## Teacher forcing

See [Seq2Seq](https://www.github.com/farizrahman4u/seq2seq)


## LSTM and GRU

Recurrent Shop comes with `LSTMCell` and `GRUCell` built-in, which can be added to RecurrentContainers using the same API discussed above.

## Finalizing your model

Once your `RecurrentContainer` is ready, you can add it to a `Sequential` model, or call it using functional API like any other layer:

```python
model = Sequential()
# Add layers, if any
model.add(rc)
# Add more layers, if any
model.compile(loss='mse', optimizer='sgd')
```

```python
a = Input((None, 20))
b = rc(a)
model = Model(a, b)
model.compile(loss='mse', optimizer='sgd')
```

------------------


# Installation

```shell
git clone https://www.github.com/datalogai/recurrentshop.git
cd recurrentshop
python setup.py install
```

# Contribute

Pull requests are highly welcome.

# Need help?

Create an issue, with a minimal script to reproduce the problem you are facing.

# Have questions?

Create an issue or drop me an email (fariz@datalog.ai).


