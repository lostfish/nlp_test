#! /usr/bin/env python
#encoding: utf-8

import numpy as np
import tensorflow as tf
from rnn_model import CharRNN, read_data

batch_size = 100        # Sequences per batch
num_steps = 100         # Number of sequence steps per batch
lstm_size = 512         # Size of hidden layers in LSTMs
num_layers = 2          # Number of LSTM layers
learning_rate = 0.001   # Learning rate
keep_prob = 0.5         # Dropout keep probability

def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c

def sample(checkpoint, n_samples, lstm_size, vocab, vocab_to_int, prime="The "):
    vocab_size = len(vocab_to_int)
    samples = [c for c in prime]
    model = CharRNN(vocab_size, lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            x[0,0] = vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state],feed_dict=feed)

        c = pick_top_n(preds, vocab_size)
        samples.append(vocab[c])

        for i in range(n_samples):
            x[0,0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state],feed_dict=feed)

            c = pick_top_n(preds, vocab_size)
            samples.append(vocab[c])

    return ''.join(samples)

#####
encoded, vocab, vocab_to_int = read_data('anna.txt')
checkpoint = tf.train.latest_checkpoint('checkpoints')
print checkpoint
samp = sample(checkpoint, 2000, lstm_size, vocab, vocab_to_int, prime="Far")
print(samp)
samp = sample(checkpoint, 1000, lstm_size, vocab, vocab_to_int, prime="Far")
print(samp)
