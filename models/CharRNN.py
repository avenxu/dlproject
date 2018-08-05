import time
import numpy as np
import tensorflow as tf


# Load data
with open('../data/anna.txt', 'r') as f:
    text = f.read()
# build char vacab
vocab = set(text)
# char-number mapping dictionary
vocab_to_int = {c: i for i, c in enumerate(vocab)}

int_to_vocab = dict(enumerate(vocab))

# encoding
encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

text[:100]

encoded[:100]

len(vocab)

def get_batches(arr, n_seqs, n_steps):
    """
    mini-batch processing
    :param arr: array to be split
    :param n_seqs: number of sequence in a batch
    :param n_steps: length of a sequence
    :return:
    """

    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)
    # we only keep the finished batches and ditch the rest
    arr = arr[:batch_size * n_batches]

    arr = arr.reshape((n_seqs, -1))

    for n in range(0, arr.shape[1], n_steps):
        # inputs
        x = arr[:, n:n + n_steps]
        # targets
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y


def build_inputs(num_seqs, num_steps):
    '''

    :param num_seqs:
    :param num_steps:
    :return:
    '''
    inputs = tf.placeholder(tf.int32, shape=(num_seqs, num_steps), name='inputs')
    targets = tf.placeholder(tf.int32, shape=(num_seqs, num_steps), name='targets')

    # keep_prob
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    return inputs, targets, keep_prob

def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    '''

    :param lstm_size:
    :param num_layers:
    :param batch_size:
    :param keep_prob:
    :return:
    '''

    lstm_cells = []
    for i in range(num_layers):
    # Create new LSTM cell
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        # lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

        # add dropout
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        lstm_cells.append(drop)
    # stack lstm cells
    cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)

    initial_state = cell.zero_state(batch_size, tf.float32)

    return cell, initial_state

def build_output(lstm_output, in_size, out_size):
    '''

    :param lstm_output:
    :param in_size:
    :param out_size:
    :return:
    '''

    # Concatenate the lstm output
    seq_output = tf.concat(lstm_output, axis=1)

    # reshape
    x = tf.reshape(seq_output, [-1, in_size])

    # softmax layer
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))

    print("X shape: {} and W shape: {}".format(x, softmax_w))
    logits = tf.matmul(x, softmax_w) + softmax_b

    out = tf.nn.softmax(logits, name='predictions')

    return out, logits

def build_loss(logits, targets, lstm_size, num_classes):
    '''

    :param logits:
    :param targets:
    :param lstm_size:
    :param num_classes:
    :return:
    '''

    # encode the targets
    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())

    # cross entropy
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)

    return loss

def build_optimizer(loss, learning_rate, grad_clip):
    '''

    :param loss:
    :param learning_rate:
    :param grad_clip:
    :return:
    '''
    # gradient clipping
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))

    return optimizer

class CharRNN:

    def __init__(self, num_classes, batch_size=64, num_steps=50,
                 lstm_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False):
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps
        tf.reset_default_graph()

        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)

        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)

        x_one_hot = tf.one_hot(self.inputs, num_classes)

        print(self.initial_state)
        print(x_one_hot)
        print(cell)
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state

        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)

        self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)