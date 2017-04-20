# State initialization

All states are by default initialized by zeros. If you want to use a different distribution (such as random normal) you can use the `state_initializer` argument available for both `RecurrentModel` and `RecurrentSequential`.


Here the `random_normal` distribution will be used to initialize all the 3 states of the RNN (2 from LSTM and 1 from GRU) : 

```python

rnn = RecurrentSequential(state_initializer='random_normal')
rnn.add(LSTMCell(10, input_dim=5))
rnn.add(GRUCell(10))

```



Here the `random_normal` distribution will be used to initialize the first state of the RNN. The rest will be initializer with zeros:


```python

rnn = RecurrentSequential(state_initializer=['random_normal'])
rnn.add(LSTMCell(10, input_dim=5))
rnn.add(GRUCell(10))

```
Here each state gets a different initializer. Note that we have specified the batch dimension as well.. this is because `glorot_uniform` initialization in Keras does not support symbolic shapes:

```python

rnn = RecurrentSequential(state_initializer=['random_normal', 'zeros', 'glorot_uniform'])
rnn.add(LSTMCell(10, batch_input_shape=(32, 10))
rnn.add(GRUCell(10))

```





