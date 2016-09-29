# Recurrent Shop
Framework for building complex recurrent neural networks with Keras


Ability to easily iterate over different neural network architectures is key to doing machine learning research. While deep learning libraries like [Keras](https://www.keras.io) makes it very easy to prototype new layers and models, writing custom recurrent neural networks is harder than it needs to be in almost all popular deep learning libraries available today. One key missing feature in these libraries is reusable RNN cells. Most libraries provide layers (such as LSTM, GRU etc), which can only be used as is, and not be easily embedded in a bigger RNN. Writing the RNN logic itself can be tiresome at times. For example in Keras, information about the states (shape and initial value) are provided by writing two seperate functions, `get_initial_states` and `reset_states` (for stateful version). There are many architectures whose implementation is not trivial using modern deep learning libraries, such as:

* Synchronising the states of all the layers in a RNN stack.
* Initializing the hidden state of a RNN with the output of a previous layer.
* Feeding back the output of the last layer of a RNN stack to the first layer in next time step (readout).
* Decoders : RNNs who can look at the whole of the input sequence / vector at every time step.

Recurrent shop adresses these issues by providing a set of *RNNCells*, which can be added sequentially to container called *RecurrentContainer* along with other layers such as `Dropout` and `Activation`, very similar to adding layers to `Sequential` model in Keras. The `RecurrentContainer` then behaves like a standard Keras `Recurrent` instance. In case of RNN stacks, the computation is done depth-first, which results in significant speed ups.

Writing the RNN logic itself has been simplified to a great extend. The user is only required to provide the step function and the shapes for the weights and the states. Default initialization for weights is glorot uniform. States are initialized by zeros, unless specified otherwise.

 * Writing a Simple RNN cell
 
 ```python
 
 class SimpleRNNCell(RNNCell):
 
  def build(self, input_shape):
    input_dim = input_shape[-1]
    output_dim = self.output_dim
    h = (-1, output_dim)  # -1 = batch size
    W = (input_dim, output_dim)
    U = (output_dim, output_dim)
    b = (self.output_sim,)
   
      def step(x, states, weights):
        h_tm1 = states[0]
        W, U, b = weights
        h = K.dot(x, W) + K.dot(h, U) + b

    self.step = step
    self.weights = [W, U, b]
    self.states = [h]

    super(SimpleRNNCell, self).build(input_shape)

```

* Recuurent container

```python

rc = RecurrentContainer()
rc.add(SimpleRNNCell(10, input_dim=20))
rc.add(Activation('tanh'))
```

* Stacking RNN cells
```python

rc = RecurrentContainer()
rc.add(SimpleRNNCell(10, input_dim=20))
rc.add(SimpleRNNCell(10))
rc.add(SimpleRNNCell(10))
rc.add(SimpleRNNCell(10))
rc.add(Activation('tanh'))

```

* State sunchronization

```python
# All cells will use the same state(s)

rc = RecurrentContainer(state_sync=True)
rc.add(SimpleRNNCell(10, input_dim=20))
rc.add(SimpleRNNCell(10))
rc.add(SimpleRNNCell(10))
rc.add(SimpleRNNCell(10))
rc.add(Activation('tanh'))
```

* Readout

```python
# Output of the final layer in the previous time step is available to the first layer(added to the input by default)

rc = RecurrentContainer(readout=True)
rc.add(SimpleRNNCell(10, input_dim=20))
rc.add(SimpleRNNCell(10))
rc.add(SimpleRNNCell(10))
rc.add(SimpleRNNCell(10))
rc.add(Activation('tanh'))
```

* Decoder

```python
# Here we decode a vector into a sequence of vectors. The input could also be a sequence, such as in the case of Attention models, where the whole input sequence is available to the RNN at every time step

# In this case, input to rc is a 2d vector, not a sequence

rc = RecurrentContainer(decode=True, output_length=10)
rc.add(SimpleRNNCell(10, input_dim=20))
```

Once your `RecurrentContainer` is ready, you can add it to a `Sequential` model, or call it using functional API like any other layer:

```python
model = Sequential()
model.add(rc)
model.compile(loss='mse', optimizer='sgd')
```

```python
a = Input()
b = rc(a)
model = Model(a, b)
model.compile(loss='mse', optimizer='sgd')
```

# Installation

```
git clone https://www.github.com/datalogai/recurrentshop.git
cd recurrentshop
python setup.py installation
```

# Contribute

Pull requests are highly welcome.

# Need help?

Create an issue, with a minimal script to reproduce the problem you are facing.

# Have questions?

Create an issue or drop me an email (fariz@datalog.ai).


