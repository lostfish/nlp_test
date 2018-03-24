#! /usr/bin/env python
#encoding: utf-8

import time
from collections import namedtuple
import numpy as np
import tensorflow as tf

def read_data(input_file):
    '''Read the novel text file
    '''
    with open(input_file) as f:
        text = f.read()
    vocab = sorted(set(text))
    vocab_to_int = {c:i for i,c in enumerate(vocab)}
    encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)
    #print len(encoded), len(vocab)
    return encoded, vocab, vocab_to_int

def get_batches(arr, batch_size, n_steps):
    '''Create a generator that return batches of inputs and targets

    Args:
        arr: an numpy array of chars
        batch_size: batch size
        n_steps: step number of rnn model
    '''
    chars_per_batch = batch_size * n_steps
    n_batches = len(arr)//chars_per_batch
    arr = arr[:n_batches*chars_per_batch]
    arr = arr.reshape([batch_size, -1]) #note this
    for n in range(0, arr.shape[1], n_steps):
        x = arr[:,n:n+n_steps]
        # tagets, shifted by one
        y_temp = arr[:,n+1:n+n_steps+1]
        # in case hit the last column
        y = np.zeros(x.shape, x.dtype)
        y[:,:y_temp.shape[1]] = y_temp
        yield x, y

def build_inputs(batch_size, n_steps):
    '''Define placeholders for inputs, targets, and dropout

    Args:
        batch_size: batch size
        n_steps: step number of rnn model
    '''
    inputs = tf.placeholder(tf.int32, shape=[batch_size, n_steps], name="inputs")
    targets = tf.placeholder(tf.int32, shape=[batch_size, n_steps], name="targets")

    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    return inputs, targets, keep_prob

def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    '''Build LSTM model
    '''
    def build_cell(lstm_size, keep_prob):
        '''
        '''
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop

    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([build_cell(lstm_size, keep_prob) for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)
    return cell, initial_state

def build_output(lstm_outputs, in_size, out_size):
    '''Build a softmax layer, return the softmax output and logits

    Args:
        lstm_outputs: lstm model outputs
        in_size: input size of the softmax layer
        out_size: output size of the softmax layer
    '''
    # batch_size*n_steps*lstm_size -> n_step*(batch_size*lstm_size)
    seq_output = tf.concat(lstm_outputs, 1)
    x = tf.reshape(seq_output, [-1, in_size])
    with tf.variable_scope("softmax"):
        softmax_w = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))
    logits = tf.matmul(x, softmax_w) + softmax_b
    out = tf.nn.softmax(logits, name="predictions")
    return out, logits

def build_loss(logits, targets, lstm_size, num_classes):
    '''Calculate the loss from the logits and the targets
    '''
    # batch_size * n_steps * num_classes
    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels = y_reshaped)
    loss = tf.reduce_mean(loss)
    return loss

def build_optimizer(loss, learning_rate, grad_clip):
    '''Build optmizer for training, using gradient clipping
    '''
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))

    return optimizer

class CharRNN():
    def __init__(self, num_classes, batch_size=64, num_steps=50,
            lstm_size=128, num_layers=2, learning_rate=0.001,
            grad_clip=5, sampling=False):
        if sampling == True:
            batch_size, num_steps = 1,1
        else:
            batch_size, num_steps = batch_size, num_steps

        tf.reset_default_graph()

        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)

        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)

        x_one_hot = tf.one_hot(self.inputs, num_classes)

        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state

        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)

        self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)

if __name__ == '__main__':
    encoded, vocab, vocab_to_int = read_data('anna.txt')
    batches = get_batches(encoded, 5, 3)
    x, y = next(batches)
    print(x)
    print(y)
