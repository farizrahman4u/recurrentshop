# State Sync

In a `RecurrentSequential`, each of the cells are given a seperate state space. For e.g:

```python
rnn = RecurrentSequential()
rnn.add(GRUCell(10, input_dim=5))
rnn.add(Dense(5))
rnn.add(LSTMCell(10))
print(rnn.num_states)  # >> 3
```

* GRUCell : 1 state
* Dense : Not an RNNCell, so 0 states
* LSTMCell : 2 states
* Total : 3

Now with state sync you can have a common state space for all the cells in a `RecurrentSequential`.
For this, all `RNNCell`s in the `RecurrentContainer` should be state homogeneous, i.e, they should all have the same number of states and corresponding states should have same shapes. Which means you can not have both `LSTMCell` and `GRUCell` in a state synced `RecurrentSequential` because `LSTMCell`s have 2 states wheareas `GRUCell`s have only 1 state.

```python
rnn = RecurrentSequential(state_sync=True)
rnn.add(LSTMCell(10, input_dim=5))
rnn.add(LSTMCell(10))
rnn.add(LSTMCell(10))
rnn.add(LSTMCell(10))
print(rnn.num_states)  # >> 2
```

All the `LSTMCell`s share the same set of 2 states.
