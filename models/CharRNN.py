import numpy as np
import tensorflow as tf
import os.path
# code reference: https://github.com/NELSONZHAO/zhihu/tree/master/anna_lstm

# Load data
abstracts = []
text = []
vocab = None

with open('../data/2017abstract.txt', 'r') as f:
    text = f.read()
vocab = sorted(set(text))
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
#
# subs = np.loadtxt('../data/2017subject.txt')
#
# data = []
# encoded = []
# encoded_file_exist = os.path.isfile('../data/2017encoded_file.txt')
# encodedWithSub_file_exist = os.path.isfile('../data/2017encoded_WithSub_file.txt')
# if not encoded_file_exist:
#     for i in range(len(abstracts)):
#         encoded = np.array([vocab_to_int[c] for c in abstracts[i]], dtype=np.int32)
#         encodedWithSub = []
#         for j in range(len(encoded)):
#             # concat each char and subject
#             encodedWithSub.append(np.append(encoded[j], subs[i]))
#         data.append(encodedWithSub)
#
#     # f1 = open("../data/2017encoded_file.txt", "w")
#     # f1.write(encoded)
#     # f2 = open("../data/2017data_file.txt", "w")
#     # f2.write(data)
# else:
#     f1 = open("../data/2017encoded_file.txt", "r")
#     encoded = f1.read()
#     f2 = open("../data/2017data_file.txt", "r")
#     data = f2.read()
#
# merged_file_exist = os.path.isfile('../data/mergeddata2017.txt')
# if not merged_file_exist:
#     print("file not exist")
#     merged_data = []
#     for sublist in data:
#         for item in sublist:
#             merged_data.append(item)
#
#     merged_data_matrix = np.array(merged_data)
#     np.savetxt("../data/mergeddata2017.txt",merged_data_matrix,fmt='%i')
# else:
#     print("file exists")
#     merged_data_matrix = np.loadtxt("../data/mergeddata2017.txt", dtype=np.int32)
#
# text[:100]
#
# encoded[:100]
#
# len(vocab)

def get_batches(arr, n_seqs, n_steps, feature_size):
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

    arr = arr.reshape((-1, n_seqs, n_steps, feature_size))

    for n in range(arr.shape[0]):
        # inputs
        x = np.array(arr[n])
        # targets
        y = np.zeros(shape=(x.shape[0], x.shape[1]))
        y[:, :-1], y[ :, -1] = x[:, 1:, 0], x[:, 0, 0]
        yield x, y


# """Testing"""
# batches = get_batches(merged_data_matrix, 10, 50, 9)
# x, y = next(batches)
#
# print('x\n', x[:10, :10])
# print('\ny\n', y[:10, :10])

def build_inputs(num_seqs, num_steps, feature_size):
    '''

    :param num_seqs:
    :param num_steps:
    :return:
    '''
    inputs = tf.placeholder(tf.int32, shape=(num_seqs, num_steps), name='inputs')
    subject = tf.placeholder(tf.int32, shape=(num_seqs, num_steps, 8), name='inputs')

    targets = tf.placeholder(tf.int32, shape=(num_seqs, num_steps), name='targets')

    # keep_prob
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    return inputs, targets, subject, keep_prob

def build_lstm(lstm_size, num_layers, batch_size, feature_size, keep_prob):
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
                 grad_clip=5, sampling=False, feature_size = 9):
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps
        tf.reset_default_graph()

        self.inputs, self.targets, self.subject, self.keep_prob = build_inputs(batch_size, num_steps, feature_size)

        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, feature_size, self.keep_prob)

        x_one_hot = tf.one_hot(self.inputs, num_classes, dtype=tf.int32)
        print(self.initial_state)
        # print(x_one_hot)
        print(cell)

        self.subject = tf.cast(self.subject, dtype=tf.int32)
        self.new_inputs = tf.concat((x_one_hot, self.subject), axis=-1)
        self.new_inputs = tf.cast(self.new_inputs, dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(cell, self.new_inputs, initial_state=self.initial_state)
        self.final_state = state

        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)

        self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)