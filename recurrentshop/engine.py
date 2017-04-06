from keras.layers import *
from keras.models import Model
from backend import rnn, learning_phase_scope



def _to_list(x):
    if type(x) is not list:
        x = [x]
    return x


def _get_cells():
    cells = {}
    cells['SimpleRNNCell'] = SimpleRNNCell
    cells['LSTMCell'] = LSTMCell
    cells['GRUCell'] = GRUCell
    return cells


def _is_rnn_cell(cell):
    return issubclass(cell.__class__, RNNCell)


def _is_all_none(iterable_or_element):
    if not isinstance(iterable_or_element, (list, tuple)):
        iterable = [iterable_or_element]
    else:
        iterable = iterable_or_element
    for element in iterable:
        if element is not None:
            return False
    return True


def _collect_input_shape(input_tensors):
    input_tensors = _to_list(input_tensors)
    shapes = []
    for x in input_tensors:
        try:
            shapes.append(K.int_shape(x))
        except TypeError:
            shapes.append(None)
    if len(shapes) == 1:
        return shapes[0]
    return shapes


_optional_key = '_optional'



class RNNCell(Layer):

    def __init__(self, output_dim=None, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.output_dim = output_dim
        if 'batch_input_shape' in kwargs:
            self.model = self.build_model(kwargs['batch_input_shape'])
        elif 'input_shape' in kwargs:
            self.model = self.build_model((None,) + kwargs['input_shape'])
        super(RNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        if type(input_shape) is list:
            self.input_spec = [InputSpec(shape=shape) for shape in input_shape]
            self.model = self.build_model(input_shape[0])
        else:
            self.model = self.build_model(input_shape)
            self.input_spec = [InputSpec(shape=shape) for shape in _to_list(self.model.input_shape)]

    def build_model(self, input_shape):
        raise Exception(NotImplemented)

    @property
    def num_states(self):
        model_input = self.model.input
        if type(model_input) is list:
            return len(model_input[1:])
        else:
            return 0

    @property
    def state_shape(self):
        model_input = self.model.input
        if type(model_input) is list:
            if len(model_input) == 2:
                return K.int_shape(model_input[1])
            else:
                return map(K.int_shape, model_input[1:])
        else:
            return None

    def compute_output_shape(self, input_shape):
        model_inputs = self.model.input
        if type(model_inputs) is list and type(input_shape) is not list:
            input_shape = [input_shape] + map(K.int_shape, model.input[1:])
        return self.model.compute_output_shape(input_shape)

    def call(self, inputs, learning=None):
        return self.model.call(inputs)

    def get_layer(self, **kwargs):
        input_shape = self.model.input_shape
        if type(input_shape) is list:
            state_shapes = input_shape[1:]
            input_shape = input_shape[0]
        else:
            state_shapes = []
        input = Input(batch_shape=input_shape)
        initial_states = [Input(batch_shape=shape) for shape in state_shapes]
        output = self.model([input] + initial_states)
        if type(output) is list:
            final_states = output[1:]
            output = output[0]
        else:
            final_states = []
        return RecurrentModel(input=input, output=output, initial_states=initial_states, final_states=final_states, **kwargs)

    @property
    def updates(self):
        return self.model.updates

    def add_update(self, updates, inputs=None):
        self.model.add_update(updates, inputs)
    
    @property
    def uses_learning_phase(self):
        return self.model.uses_learning_phase
    
    @property
    def _per_input_losses(self):
        return getattr(self.model, '_per_input_losses', {})

    @property
    def losses(self):
        return self.losses
    
    def add_loss(self, losses, inputs=None):
        self.model.add_loss(losses, inputs)

    @property
    def constraints(self):
        return self.model.constraints
    
    @property
    def trainable_weights(self):
        return self.model.trainable_weights

    @property
    def non_trainable_weights(self):
        return self.model.non_trainable_weights
    
    def get_losses_for(self, inputs):
        return self.model.get_losses_for(inputs)

    def get_updates_for(self, inputs):
        return self.model.get_updates_for(inputs)

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def get_config(self):
        config = {'output_dim': self.output_dim}
        base_config = super(RNNcell, self).get_config()
        config.update(base_config)
        return config

    def compute_mask(self, inputs, mask=None):
        model_output = self.model.output
        if type(model_output) is list:
            return [None] * len(model_output)
        else:
            return None


class RNNCellFromModel(RNNCell):

    def __init__(self, model, **kwargs):
        self.model = model
        self.input_spec = [Input(batch_shape=shape) for shape in _to_list(model.input_shape)]
        self.build_model = lambda _: model
        super(RNNCellFromModel, self).__init__(batch_input_shape=model.input_shape, **kwargs)

    def get_config(self):
        config = super(RNNCellFromModel, self).get_config()
        config['model_config'] = self.model.get_config()
        return config

    def from_config(cls, config, custom_objects={}):
        if type(custom_objects) is list:
            custom_objects = {obj.__name__: obj for obj in custom_objects}
        custom_objects.update(_get_cells())
        model_config = config.pop('model_config')
        model = Model.from_config(model_config, custom_objects)
        return cls(model, **config)


class RecurrentModel(Recurrent):

    def __init__(self, input, output, initial_states=None, final_states=None, readout_input=None, teacher_force=False, decode=False, output_length=None, return_states=False, **kwargs):
        inputs = [input]
        outputs = [output]
        state_spec = None
        if initial_states:
            if type(initial_states) not in [list, tuple]:
                initial_states = [initial_states]
            state_spec = [InputSpec(shape=K.int_shape(state)) for state in initial_states]
            if not final_states:
                raise Exception('Missing argument : final_states')
            else:
                self.states = [None] * len(initial_states)
            inputs += initial_states
        else:
            self.states = []
            state_spec = []

        if final_states:
            if type(final_states) not in [list, tuple]:
                final_states = [final_states]
            assert len(initial_states) == len(final_states), 'initial_states and final_states should have same number of tensors.'
            if not initial_states:
                raise Exception('Missing argument : initial_states')
            outputs += final_states
        self.decode = decode
        self.output_length = output_length
        if decode:
            if output_length is None:
                raise Exception('output_length should be specified for decoder')
            kwargs['return_sequences'] = True
        self.return_states = return_states
        if readout_input:
            self.readout = True
            state_spec += [Input(batch_shape=K.int_shape(outputs[0]))]
            self.states += [None]
        else:
            self.readout = False
        if self.teacher_force and not self.readout:
            raise Exception('Readout should be enabled for teacher forcing.')
        self.teacher_force = teacher_force
        self.model = Model(inputs, outputs)
        super(RecurrentModel, self).__init__(**kwargs)
        input_shape = list(K.int_shape(input))
        if not decode:
            input_shape.insert(1, None)
        self.input_spec = InputSpec(shape=tuple(input_shape))
        self.state_spec = state_spec
        self._optional_input_placeholders = {}


    def build(self, input_shape):
        if type(input_shape) is list:
            input_shape = input_shape[0]
        if not self.decode:
            input_length = input_shape[1]
            if input_length is not None:
                input_shape = list(self.input_spec.shape)
                input_shape[1] = input_length
                input_shape = tuple(input_shape)
                self.input_spec = InputSpec(shape=input_shape)
        if type(self.model.input) is list:
            model_input_shape = self.model.input_shape[0]
        else:
            model_input_shape = self.model.input_shape
        if not self.decode:
            input_shape = input_shape[:1] + input_shape[2:]
        for i, j in zip(input_shape, model_input_shape):
            if i is not None and j is not None and i != j:
                raise Exception('Model expected input with shape ' + str(model_input_shape) +
                    '. Received input with shape ' + str(input_shape))
        if self.stateful:
            self.reset_states()

    def step(self, inputs, states):
        if self.decode:
            model_input = list(states)
        else:
            model_input = [inputs] + list(states)
        shapes = []
        for x in model_input:
            if hasattr(x, '_keras_shape'):
                shapes.append(x._keras_shape)
                del x._keras_shape  # Else keras internals will get messed up.
        model_output = _to_list(self.model.call(model_input))
        for x, s in zip(model_input, shapes):
            setattr(x, '_keras_shape', s)
        if self.decode:
            model_output.insert(1, model_input[0])
        for tensor in model_output:
            tensor._uses_learning_phase = self.uses_learning_phase
        states = model_output[1:]
        output = model_output[0]
        if self.readout:
            states += [output]
        return output, states

    def get_initial_state(self, inputs):
        if type(self.model.input) is not list:
            return []
        try:
            batch_size = K.int_shape(inputs)[0]
        except:
            batch_size = None
        state_shapes = map(K.int_shape, self.model.input[1:])
        states = []
        for shape in state_shapes:
            if None in shape[1:]:
                raise Exception('Only the batch dimension of a state can be left unspecified. Got state with shape ' + str(shape))
            if shape[0] is None:
                ndim = K.ndim(inputs)
                z = K.zeros_like(inputs)
                slices = [slice(None)] + [0] * (ndim - 1)
                z = z[slices]  # (batch_size,)
                state_ndim = len(shape)
                z = K.reshape(z, (-1,) + (1,) * (state_ndim - 1))
                z = K.tile(z, (1,) + tuple(shape[1:]))
                states.append(z)
            else:
                states.append(K.zeros(shape))
        return states

    def reset_states(self, states_value=None):
        if len(self.states) == 0:
            return
        if not self.stateful:
            raise AttributeError('Layer must be stateful.')
        if not hasattr(self, 'states') or self.states[0] is None:
            state_shapes = map(K.int_shape, self.model.input[1:])
            self.states = map(K.zeros, state_shapes)

        if states_value is not None:
            if type(states_value) not in (list, tuple):
                states_value = [states_value] * len(self.states)
            assert len(states_value) == len(self.states), 'Your RNN has ' + str(len(self.states)) + ' states, but was provided ' + str(len(states_value)) + ' state values.'
            if 'numpy' not in type(states_value[0]):
                states_value = map(np.array, states_value)
            if states_value[0].shape == tuple():
                for state, val in zip(self.states, states_value):
                    K.set_value(state, K.get_value(state) * 0. + val)
            else:
                for state, val in zip(self.states, states_value):
                    K.set_value(state, val)

    def _get_optional_input_placeholder(self, name=None, num=1):
        if name:
            if name not in self._optional_input_placeholders:
                if num > 1:
                    self._optional_input_placeholders[name] = [self._get_optional_input_placeholder() for _ in range(num)]
                else: 
                    self._optional_input_placeholders[name] = self._get_optional_input_placeholder()
            return self._optional_input_placeholders[name]
        if num == 1:
            optional_input_placeholder = Input(batch_shape=(None,))
            optional_input_placeholder.name += '_optional'
            return optional_input_placeholder
        else:
            y = []
            for _ in range(num):
                optional_input_placeholder = Input(batch_shape=(None,))
                optional_input_placeholder.name += '_optional'
                y.append(optional_input_placeholder)
            return y            


    def _is_optional_input_placeholder(self, x):
        return x.name[-9:] == '_optional'



    def __call__(self, inputs, initial_state=None, initial_readout=None, ground_truth=None, **kwargs):
        inputs = _to_list(inputs)
        if len(inputs) == 1:
            if initial_state:
                if type(initial_state) is list:
                    inputs += initial_state
                else:
                    inputs.append(initial_state)
            else:
                initial_state = self._get_optional_input_placeholder('initial_state', len(self.state))
                inputs += _to_list(initial_state)
            if not initial_readout:
                initial_readout = self._get_optional_input_placeholder('initial_readout')
            inputs.append(initial_readout)
            if not ground_truth:
                ground_truth = self._get_optional_input_placeholder('ground_truth')
            inputs.append(ground_truth)
        assert len(inputs) == len(self.states) + 3
        with K.name_scope(self.name):
            if not self.built:
                self.build(K.int_shape(inputs[0]))
                if self._initial_weights is not None:
                    self.set_weights(self._initial_weights)
                    del self._initial_weights
                    self._initial_weights = None
            previous_mask = _collect_previous_mask(inputs[:1])
            if not _is_all_none(previous_mask):
                if 'mask' in inspect.getargspec(self.call).args:
                    if 'mask' not in kwargs:
                        kwargs['mask'] = previous_mask
            input_shape = _collect_input_shape(inputs)
            output = self.call(inputs, initial_states, initial_readout, ground_truth, **kwargs)
            output_mask = self.compute_mask(inputs[0], previous_mask)

            # Infering the output shape is only relevant for Theano.
            output_shape = self.compute_output_shape(input_shape[0])

            self._add_inbound_node(input_tensors=inputs, output_tensors=output,
                                   input_masks=previous_mask, output_masks=output_mask,
                                   input_shapes=input_shape, output_shapes=output_shape,
                                   arguments=kwargs)

            # Apply activity regularizer if any:
            if hasattr(self, 'activity_regularizer') and self.activity_regularizer is not None:
                regularization_losses = [self.activity_regularizer(x) for x in _to_list(output)]
                self.add_loss(regularization_losses, _to_list(inputs))
        return output



    def call(self, inputs, mask=None, initial_state=None, initial_readout=None, ground_truth=None, training=None):
        # input shape: `(samples, time (padded with zeros), input_dim)`
        # note that the .build() method of subclasses MUST define
        # self.input_spec and self.state_spec with complete input shapes.
        if initial_state is not None:
            if not isinstance(initial_state, (list, tuple)):
                initial_states = [initial_state]
            else:
                initial_states = list(initial_state)
        if isinstance(inputs, list):
            initial_states = inputs[1:]
            inputs = inputs[0]
        elif self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_state(inputs)

        if len(initial_states) != len(self.states):
            raise ValueError('Layer has ' + str(len(self.states)) +
                             ' states but was passed ' +
                             str(len(initial_states)) +
                             ' initial states.')
        input_shape = K.int_shape(inputs)
        if self.unroll and input_shape[1] is None:
            raise ValueError('Cannot unroll a RNN if the '
                             'time dimension is undefined. \n'
                             '- If using a Sequential model, '
                             'specify the time dimension by passing '
                             'an `input_shape` or `batch_input_shape` '
                             'argument to your first layer. If your '
                             'first layer is an Embedding, you can '
                             'also use the `input_length` argument.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a `shape` '
                             'or `batch_shape` argument to your Input layer.')
        constants = self.get_constants(inputs, training=None)
        preprocessed_input = self.preprocess_input(inputs, training=None)
        if self.decode:
            initial_states.insert(0, inputs)
            preprocessed_input = K.zeros((1, self.output_length, 1))
            input_length = self.output_length
        else:
            input_length = input_shape[1]
        if self.readout:
            if not initial_readout:
                ndim = K.ndim(inputs)
                initial_readout = K.zeros_like(inputs)
                slices = [slice(None)] + [0] * (ndim - 1)
                initial_readout = initial_readout[slices]  # (batch_size,)
                output_ndim = K.int_shape(_to_list(model.output)[0])
                initial_readout = K.reshape(initial_readout, (-1,) + (1,) * (output_ndim - 1))
                initial_readout = K.tile(initial_readout, (1,) + tuple(shape[1:]))
                initial_states.append(initial_readout)
        if self.uses_learning_phase:
            with learning_phase_scope(0):
                last_output_test, outputs_test, states_test, updates = rnn(self.step,
                                                 preprocessed_input,
                                                 initial_states,
                                                 go_backwards=self.go_backwards,
                                                 mask=mask,
                                                 constants=constants,
                                                 unroll=self.unroll,
                                                 input_length=input_length)
            with learning_phase_scope(1):
                last_output_train, outputs_train, states_train, updates = rnn(self.step,
                                                 preprocessed_input,
                                                 initial_states,
                                                 go_backwards=self.go_backwards,
                                                 mask=mask,
                                                 constants=constants,
                                                 unroll=self.unroll,
                                                 input_length=input_length)

            last_output = K.in_train_phase(last_output_train, last_output_test, training=training)
            outputs = K.in_train_phase(outputs_train, outputs_test, training=training)
            states = []
            for state_train, state_test in zip(states_train, states_test):
                states.append(K.in_train_phase(state_train, state_test, training=training))

        else:
            last_output, outputs, states, updates = rnn(self.step,
                                                 preprocessed_input,
                                                 initial_states,
                                                 go_backwards=self.go_backwards,
                                                 mask=mask,
                                                 constants=constants,
                                                 unroll=self.unroll,
                                                 input_length=input_length)
        if self.decode:
            states.pop(0)
        if self.readout:
            states.pop(-1)
        if len(updates) > 0:
            self.add_update(updates)
        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_update(updates, inputs)

        # Properly set learning phase
        if 0 < self.dropout + self.recurrent_dropout:
            last_output._uses_learning_phase = True
            outputs._uses_learning_phase = True

        if self.return_sequences:
            y = outputs
        else:
            y = last_output
        if self.return_states:
            return [y] + states
        else:
            return y

    @property
    def updates(self):
        return self.model.updates


    def add_update(self, updates, inputs=None):
        self.model.add_update(updates, inputs)
    
    @property
    def uses_learning_phase(self):
        return self.model.uses_learning_phase
    
    @property
    def _per_input_losses(self):
        return getattr(self.model, '_per_input_losses', {})

    @property
    def losses(self):
        return self.losses
    
    def add_loss(self, losses, inputs=None):
        self.model.add_loss(losses, inputs)

    @property
    def constraints(self):
        return self.model.constraints
    
    @property
    def trainable_weights(self):
        return self.model.trainable_weights

    @property
    def non_trainable_weights(self):
        return self.model.non_trainable_weights
    
    def get_losses_for(self, inputs):
        return self.model.get_losses_for(inputs)

    def get_updates_for(self, inputs):
        return self.model.get_updates_for(inputs)

    def _remove_time_dim(self, shape):
        return shape[:1] + shape[2:]

    def compute_output_shape(self, input_shape):
        if not self.decode:
            if type(input_shape) is list:
                input_shape[0] = self._remove_time_dim(input_shape[0])
            else:
                input_shape = self._remove_time_dim(input_shape)
        if len(self.states) > 0 and (type(input_shape) is not list or len(input_shape) == 1):
            input_shape = _to_list(input_shape) + [K.int_shape(state) for state in self.model.input[1:]]
        output_shape = self.model.compute_output_shape(input_shape)
        if type(output_shape) is list:
            output_shape = output_shape[0]
        if self.return_sequences:
            if self.decode:
                output_shape = output_shape[:1] + (self.output_length,) + output_shape[1:] 
            else:
                output_shape = output_shape[:1] + (self.input_spec.shape[1],) + output_shape[1:]
        if self.return_states and len(self.states) > 0:
            output_shape = [output_shape] + list(map(K.int_shape, self.model.output[1:]))
        return output_shape

    def compute_mask(self, input, input_mask=None):
        mask = input_mask[0] if type(input_mask) is list else input_mask
        mask = mask if self.return_sequences else None
        mask = [mask] + [None] * len(self.states) if self.return_states else mask
        return mask

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def get_config(self):
        config = {'model_config': self.model.get_config(),
                  'decode': self.decode,
                  'output_length': self.output_length,
                  'return_states': self.return_states}
        base_config = super(RecurrentModel, self).get_config()
        config.update(base_config)
        return config

    def from_config(cls, config, custom_objects={}):
        if type(custom_objects) is list:
            custom_objects = {obj.__name__: obj for obj in custom_objects}
        custom_objects.update(_get_cells())
        config = config.copy()
        model_config = config.pop('model_config')
        model = Model.from_config(model_config, custom_objects)
        if type(model.input) is list:
            input = model.input[0]
            initial_states = model.input[1:]
        else:
            input = model.input
            initial_states = None
        if type(model.output) is list:
            output = model.output[0]
            final_states = model.output[1:]
        else:
            output  = model.output
            final_states = None
        return cls(input, output, initial_states, final_states, **config)

    def get_cell(self, **kwargs):
        return RNNCellFromModel(self.model, **kwargs)


class RecurrentSequential(RecurrentModel):

    def __init__(self, state_sync=False, decode=False, output_length=None, return_states=False, **kwargs):
        self.state_sync = state_sync
        self.cells = []
        if decode and output_length is None:
            raise Exception('output_length should be specified for decoder')
        self.decode = decode
        self.output_length = output_length
        if decode:
            if output_length is None:
                raise Exception('output_length should be specified for decoder')
            kwargs['return_sequences'] = True
        self.return_states = return_states
        super(RecurrentModel, self).__init__(**kwargs)

    def add(self, cell):
        self.cells.append(cell)
        if len(self.cells) == 1:
            cell_input_shape = cell.batch_input_shape
            if set(map(type, list(set(cell_input_shape) - set([None])))) != set([int]):
                cell_input_shape = cell_input_shape[0]
            if self.decode:
                self.input_spec = InputSpec(shape=cell_input_shape)
            else:
                self.input_spec = InputSpec(shape=cell_input_shape[:1] + (None,) + cell_input_shape[1:])

    def build(self, input_shape):
        if hasattr(self, 'model'):
            del self.model
        if self.state_sync:
            if type(input_shape) is list:
                x_shape = input_shape[0]
                if not self.decode:
                    input_length = x_shape.pop(1)
                    if input_length is not None:
                        shape = list(self.input_spec.shape)
                        shape[1] = input_length
                        self.input_spec = InputSpec(shape=tuple(shape))
                input = Input(batch_shape=x_shape)
                initial_states = [Input(batch_shape=shape) for shape in input_shape[1:]]
            else:
                if not self.decode:
                    input_length = input_shape[1]
                    if input_length is not None:
                        shape = list(self.input_spec.shape)
                        shape[1] = input_length
                        self.input_spec = InputSpec(shape=tuple(shape))
                    input = Input(batch_shape=input_shape[:1] + input_shape[2:])
                else:
                    input = Input(batch_shape=input_shape)
                initial_states = []
            output = input
            final_states = initial_states[:]
            for cell in self.cells:
                if _is_rnn_cell(cell):
                    if not initial_states:
                        cell.build(K.int_shape(output))
                        initial_states = [Input(batch_shape=shape) for shape in _to_list(cell.state_shape)]
                        final_states = initial_states[:]
                    cell_out = cell([output] + final_states)
                    if type(cell_out) is not list:
                        cell_out = [cell_out]
                    output = cell_out[0]
                    final_states = cell_out[1:]
                else:
                    output = cell(output)
        else:
            if type(input_shape) is list:
                x_shape = input_shape[0]
                if not self.decode:
                    input_length = x_shape.pop(1)
                    if input_length is not None:
                        shape = list(self.input_spec.shape)
                        shape[1] = input_length
                        self.input_spec = InputSpec(shape=tuple(shape))
                input = Input(batch_shape=x_shape)
                initial_states = [Input(batch_shape=shape) for shape in input_shape[1:]]
                output = input
                final_states = []
                for cell in self.cells:
                    if _is_rnn_cell(cell):
                        cell_initial_states = initial_states[len(final_states) : len(final_states) + cell.num_states]
                        cell_in = [output] + cell_initial_states
                        cell_out = _to_list(cell(cell_in))
                        output = cell_out[0]
                        final_states += cell_out[1:]
                    else:
                        output = cell(output)
            else:
                if not self.decode:
                    input_length = input_shape[1]
                    if input_length is not None:
                        shape = list(self.input_spec.shape)
                        shape[1] = input_length
                        self.input_spec = InputSpec(shape=tuple(shape))
                    input = Input(batch_shape=input_shape[:1] + input_shape[2:])
                else:
                    input = Input(batch_shape=input_shape)
                output = input
                initial_states = []
                final_states = []
                for cell in self.cells:
                    if _is_rnn_cell(cell):
                        cell.build(K.int_shape(output))
                        state_inputs = [Input(batch_shape=shape) for shape in _to_list(cell.state_shape)]
                        initial_states += state_inputs
                        cell_in = [output] + state_inputs
                        cell_out = _to_list(cell(cell_in))
                        output = cell_out[0]
                        final_states += cell_out[1:]
                    else:
                        output = cell(output)
        self.model = Model([input] + initial_states, [output] + final_states)
        self.states = [None] * len(initial_states)
        super(RecurrentSequential, self).build(input_shape)

    def get_config(self):
        config = {'state_sync': self.state_sync}
        base_config = super(RecurrentSequential, self).get_config()
        config.update(base_config)
        return config

# Legacy
RecurrentContainer = RecurrentSequential
