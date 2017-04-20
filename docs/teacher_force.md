# Teacher forcing

One issue you might face with RNNs with readout is the accumulation of error. On every time step, you feed in the output of the previous time step; but during training, the output of the previous time step (which is usually a probability distribution for a class label) will not be accurate initially. Over long sequences, this error might accumulate and lead to poor performance.

You can rectify this issue by teacher forcing, where you feed the ground truth at the previous time step to the current time step as readout (instead of the prediction at previous time step). 


```python

rnn = RecurrentSequential(readout='add', teacher_force=True, return_sequences=True)
rnn.add(LSTMCell(10, input_dim=10))
rnn.add(LSTMCell(10))
rnn.add(LSTMCell(10))
rnn.add(Activation('softmax'))


x = Input((7, 10))
y_true = Input((7, 10))  # This is where you feed the ground truth values

y = rnn(x, ground_truth=y_true)

model = Model([x, y_true], y)

model.compile(loss='categorical_crossentropy', optimizer='sgd')

# Training

X_true = np.random.random((32, 7, 10))
Y_true = np.random.random((32, 7, 10))

model.fit([X_true, Y_true], Y_true)  # Note that Y_true is part of both input and output


# Prediction

X = np.random.random((32, 7, 10))

model.predict(X)  # >> Error! the graph still has an input for ground truth.. 

zeros = np.zeros((32, 7, 10)) # Need not be zeros.. any array of same shape would do

model.predict([X, zeros])
```

