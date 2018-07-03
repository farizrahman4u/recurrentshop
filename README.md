# Recurrent Shop
Framework for building complex recurrent neural networks with Keras


Ability to easily iterate over different neural network architectures is key to doing machine learning research. While deep learning libraries like [Keras](https://www.keras.io) makes it very easy to prototype new layers and models, writing custom recurrent neural networks is harder than it needs to be in almost all popular deep learning libraries available today. One key missing feature in these libraries is reusable RNN cells. Most libraries provide layers (such as LSTM, GRU etc), which can only be used as is, and not be easily embedded in a bigger RNN. Writing the RNN logic itself can be tiresome at times. For example in Keras, information about the states (shape and initial value) are provided by writing two seperate functions, `get_initial_states` and `reset_states` (for stateful version). There are many architectures whose implementation is not trivial using modern deep learning libraries, such as:

* Synchronising the states of all the layers in a RNN stack.
* Feeding back the output of the last layer of a RNN stack to the first layer in next time step (readout).
* Decoders : RNNs who can look at the whole of the input sequence / vector at every time step.
* Teacher forcing : Using the ground truth at time t-1 for predicting at time t during training.
* Nested RNNs.
* Initializing states with different distributions.


Recurrent shop adresses these issues by letting the user write RNNs of arbitrary complexity using Keras's functional API. In other words, the user builds a standard Keras model which defines the logic of the RNN for a single timestep, and RecurrentShop converts this model into a `Recurrent` instance, which is capable of processing sequences.


## Writing a Simple RNN using Functional API
 
```python
# The RNN logic is written using Keras's functional API.
# Which means we use Keras layers instead of theano/tensorflow ops
from keras.layers import *
from keras.models import *
from recurrentshop import *

x_t = Input(shape=(5,)) # The input to the RNN at time t
h_tm1 = Input(shape=(10,))  # Previous hidden state

# Compute new hidden state
h_t = add([Dense(10)(x_t), Dense(10, use_bias=False)(h_tm1)])

# tanh activation
h_t = Activation('tanh')(h_t)

# Build the RNN
# RecurrentModel is a standard Keras `Recurrent` layer. 
# RecurrentModel also accepts arguments such as unroll, return_sequences etc
rnn = RecurrentModel(input=x_t, initial_states=[h_tm1], output=h_t, final_states=[h_t])
# return_sequences is False by default
# so it only returns the last h_t state

# Build a Keras Model using our RNN layer
# input dimensions are (Time_steps, Depth)
x = Input(shape=(7,5))
y = rnn(x)
model = Model(x, y)

# Run the RNN over a random sequence
# Don't forget the batch shape when calling the model!
out = model.predict(np.random.random((1, 7, 5)))
print(out.shape)#->(1,10)


# to get one output per input sequence element, set return_sequences=True
rnn2 = RecurrentModel(input=x_t, initial_states=[h_tm1], output=h_t, final_states=[h_t],return_sequences=True)

# Time_steps can also be None to allow variable Sequence Length
# Note that this is not compatible with unroll=True
x = Input(shape=(None ,5))
y = rnn2(x)
model2 = Model(x, y)

out2 = model2.predict(np.random.random((1, 7, 5)))
print(out2.shape)#->(1,7,10)

```

## RNNCells

An `RNNCell` is a layer which defines the computation of an RNN for a single timestep. It takes a list of tensors as input (`[input, state1_tm1, state2_tm1..]`) and outputs a list of tensors (`[output, state1_t, state2_t...]`). An RNNCell does not iterate over an input sequence. It works on a single time step. So the shape of the input to an `LSTMCell` would be `(batch_size, input_dim)` rather than `(batch_size, input_length, input_dim)`

RecurrentShop comes with 3 built-in RNNCells : `SimpleRNNCell`, `GRUCell`, and `LSTMCell`
There are 2 versions of each of these cells. [The basic version which is more readable](recurrentshop/basic_cells.py) which you can refer to learn how to write custom RNNCells and the [customizable and recommended version](recurrentshop/cells.py) which has more options like setting regularizers, constraints, activations etc.

An `RNNCell` can be easily converted to a Keras `Recurrent` layer:

```python
from recurrentshop.cells import LSTMCell

lstm_cell = LSTMCell(10, input_dim=5)
lstm_layer = lstm_cell.get_layer()

# get_layer accepts arguments like return_sequences, unroll etc :
lstm_layer = lstm_cell.get_layer(return_sequences=True, unroll=True)

```

## RecurrentSequential

`RecurrentSequential` is the Recurrent analog for Keras's `Sequential` model. It lets you stack RNNCells and other layers such as `Dense` and `Activation` to build a Recurrent layer:

```python
rnn = RecurrentSequential(unroll=False, return_sequences=False)
rnn.add(SimpleRNNCell(10, input_dim=5))
rnn.add(LSTMCell(12))
rnn.add(Dense(5))
rnn.add(GRU(8))

# rnn can now be used as regular Keras Recurrent layer.
```

## Nesting RecurrentSequentials

A `RecurrentSequential` (or any `RecurrentModel`)  can be converted to a cell using the `get_cell()` method. This cell can then be added to another `RecurrentSequential`.

```python
rnn1 = RecurrentSequential()
rnn1.add(....)
rnn1.add(....)

rnn1_cell = rnn1.get_cell()

rnn2 = RecurrentSequential()
rnn2.add(rnn1_cell)
rnn2.add(...)
```

## Using RNNCells in Functional API

Since an `RNNCell` is a regular Keras layer by inheritance, it can be used for building `RecurrentModel`s using functional API.

```python
from recurrentshop import *
from keras.layers import *
from keras.models import Model

input = Input((5,))
state1_tm1 = Input((10,))
state2_tm1 = Input((10,))
state3_tm1 = Input((10,))

lstm_output, state1_t, state2_t = LSTMCell(10)([input, state1_tm1, state2_tm1])
gru_output, state3_t = GRUCell(10)([input, state3_tm1])

output = add([lstm_output, gru_output])
output = Activation('tanh')(output)

rnn = RecurrentModel(input=input, initial_states=[state1_tm1, state2_tm1, state3_tm1], output=output, final_states=[state1_t, state2_t, state3_t])
```

# More features

See docs/ directory for more features.


# Installation

```shell
git clone https://www.github.com/farizrahman4u/recurrentshop.git
cd recurrentshop
python setup.py install
```

# Contribute

Pull requests are highly welcome.

# Need help?

Create an issue, with a minimal script to reproduce the problem you are facing.

# Have questions?

Create an issue or drop me an email (farizrahman4u@gmail.com).

