# Readout

Readout lets you feed the output of your RNN from the previous time step back to the current time step.

## Readout in RecurrentSequential

```python
rnn = RecurrentSequential(readout='add')
rnn.add(LSTMCell(10, input_dim=10))
rnn.add(GRUCell(10))
rnn.add(SimpleRNNCell(10))
```
The output from the previous time step will be added to the current input. Other modes available are : `mul`, `avg`, `max`. (Note : since these are elem-wise ops, output shape and input shape of the RNN should be the same.)


## Readout in RecurrentModel

In case you want to do something more complex than just merge your readout with input, you can wire things up with functional API and use `RecurrentModel` to build your RNN.

```python

x = Input((10,))
h_tm1 = Input((10,))
c_tm1 = Input((10,))
readout_input = Input((10,))


# Here, I simply add half the readout to the input.. you can do whatever you want.

readout_half = Lambda(lambda x: 0.5 * x)(readout_input)
lstms_input = add([x, readout_half])

# Deep LSTM
depth = 3

cells = [LSTMCell(10) for _ in range(depth)]

lstms_output, h, c = lstms_input, h_tm1, c_tm1


for cell in cells:
    lstms_output, h, c = cell([lstms_output, h, c])

y = lstms_output

rnn = RecurrentModel(input=x, initial_states=[h_tm1, c_tm1], output=y, final_states=[h, c], readout_input=readout_input)

```
