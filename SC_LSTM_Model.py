#coding:utf-8
import tensorflow as tf
import numpy as np
from tensorflow.nn.rnn_cell import BasicLSTMCell
from tensorflow.nn.rnn_cell import DropoutWrapper
from tensorflow.contrib.layers import fully_connected as FC
from tensorflow.nn.rnn_cell import MultiRNNCell
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.util import nest
_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"
class SC_LSTM(BasicLSTMCell):
    def __init__(self, kwd_voc_size, *args, **kwargs):
        BasicLSTMCell.__init__(self, *args, **kwargs)
        self.key_words_voc_size = kwd_voc_size
    def _linear(self,args, output_size, bias, bias_start=0.0):
  
        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")
        if not nest.is_sequence(args):
            args = [args]

        # Calculate the total size of arguments on dimension 1.
        total_arg_size = 0
        shapes = [a.get_shape() for a in args]
        for shape in shapes:
            if shape.ndims != 2:
                raise ValueError("linear is expecting 2D arguments: %s" % shapes)
            if shape[1].value is None:
                raise ValueError("linear expects shape[1] to be provided for shape %s, ""but saw %s" % (shape, shape[1]))
            else:
                total_arg_size += shape[1].value

        dtype = [a.dtype for a in args][0]

  # Now the computation.
        scope = vs.get_variable_scope()
        with vs.variable_scope(scope) as outer_scope:
            weights = vs.get_variable(_WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size], dtype=dtype)
            if len(args) == 1:
                res = math_ops.matmul(args[0], weights)
            else:
                res = math_ops.matmul(array_ops.concat(args, 1), weights)
            if not bias:
                return res
            with vs.variable_scope(outer_scope) as inner_scope:
                inner_scope.set_partitioner(None)
                biases = vs.get_variable(_BIAS_VARIABLE_NAME,[output_size],dtype=dtype,initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
        return nn_ops.bias_add(res, biases)
    def __call__(self, inputs, state, d_act, scope=None):
        """Long short-term memory cell (LSTM)."""


        with vs.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
        # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                try:
                    c, h = array_ops.split(1, 2, state)
                except:
                    c, h = array_ops.split(state, 2, 1)
            concat = self._linear([inputs, h], 4 * self._num_units,True)
            #concat = math_ops.matmul(array_ops.concat([inputs, h], 1), self._kernel)
            #concat = nn_ops.bias_add(concat, self._bias)
            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            try:
                i, j, f, o = array_ops.split(1, 4, concat)
            except:
                i, j, f, o = array_ops.split(concat, 4, 1)
            
            w_d = vs.get_variable('w_d', [self.key_words_voc_size, self._num_units])
            
            new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) *
                    self._activation(j)) + tf.tanh(tf.matmul(d_act, w_d))
            new_h = self._activation(new_c) * sigmoid(o)

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                try:
                    new_state = array_ops.concat(1, [new_c, new_h])
                except:
                    new_state = array_ops.concat([new_c, new_h], 1)
            return new_h, new_state

class SC_MultiRNNCell(MultiRNNCell):
    def __call__(self, inputs, state, d_act, scope=None):
        """Run this multi-layer cell on inputs, starting from state."""
        with vs.variable_scope(scope or type(self).__name__):  # "MultiRNNCell"
            cur_state_pos = 0
            cur_inp = inputs
            new_states = []
            outputls = []
            for i, cell in enumerate(self._cells):
                with vs.variable_scope("Cell%d" % i):
                    if self._state_is_tuple:
                        if not nest.is_sequence(state):
                            raise ValueError(
                                "Expected state to be a tuple of length %d, but received: %s"
                                % (len(self.state_size), state))
                        cur_state = state[i]
                    else:
                        cur_state = array_ops.slice(
                            state, [0, cur_state_pos], [-1, cell.state_size])
                        cur_state_pos += cell.state_size
                    cur_inp, new_state = cell(cur_inp, cur_state, d_act)
                    new_states.append(new_state)
                    outputls.append(cur_inp)
        try:
            new_states = (tuple(new_states) if self._state_is_tuple
                      else array_ops.concat(1, new_states))
            outputs = array_ops.concat(1, outputls)
        except:
            new_states = (tuple(new_states) if self._state_is_tuple
                      else array_ops.concat(new_states, 1))
            outputs = array_ops.concat(outputls, 1)
        return cur_inp, new_states, outputs

class SC_DropoutWrapper(DropoutWrapper):
    def __call__(self, inputs, state, d_act, scope=None):
        """Run the cell with the declared dropouts."""
        if (not isinstance(self._input_keep_prob, float) or
                self._input_keep_prob < 1):
            inputs = nn_ops.dropout(inputs, self._input_keep_prob, seed=self._seed)
        output, new_state = self._cell(inputs, state, d_act, scope)
        if (not isinstance(self._output_keep_prob, float) or
                self._output_keep_prob < 1):
            output = nn_ops.dropout(output, self._output_keep_prob, seed=self._seed)
        return output, new_state
