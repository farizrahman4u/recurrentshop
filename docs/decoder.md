# Decoder


Here we decode a vector into a sequence of vectors. The input could also be a sequence, such as in the case of Attention models, where the whole input sequence is available to the RNN at every time step


In this case, input to rnn is a 2d vector, not a sequence

```python
rnn = RecurrentSequential(decode=True, output_length=10)
rnn.add(SimpleRNNCell(25, input_dim=20))

x = Input((20,))
y = rnn(x)

print(K.int_shape(y))  # >> (None, 10, 25)

```
