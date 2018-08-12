import tensorflow as tf
import numpy as np
from models import CharRNN
# check checkpoints
tf.train.get_checkpoint_state('checkpoints')




def pick_top_n(preds, vocab_size, top_n=5):
    """
    pick the top n results

    preds
    vocab_size
    top_n
    """
    p = np.squeeze(preds)
    # set the rest 0
    p[np.argsort(p)[:-top_n]] = 0
    # regularization
    p = p / np.sum(p)
    # pick a random one
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


def sample(checkpoint, n_samples, lstm_size, vocab_size, prime="We", subject=[0,0,0,0,0,0,0,0]):
    """
    Sampling new text

    checkpoint
    n_sample: length of the sample
    lstm_size: number of hidden nodes
    vocab_size
    prime: start text
    """
    # convert input word to chars

    samples = [c for c in prime]
    # sampling=True means batch of size=1 x 1
    model = CharRNN.CharRNN(len(CharRNN.vocab), lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Restore session
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            # input single char
            x[0, 0] = CharRNN.vocab_to_int[c]
            subject_reshape = np.reshape(subject, (1, 1, -1)).astype(dtype=int)

            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.subject: subject_reshape,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state],
                                        feed_dict=feed)

        c = pick_top_n(preds, len(CharRNN.vocab))
        # add new predictions to the sampling
        samples.append(CharRNN.int_to_vocab[c])

        # generate new chars till the limit
        for i in range(n_samples):
            x[0, 0] = c
            subject_reshape = np.reshape(subject, (1, 1, -1)).astype(dtype=int)
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.subject: subject_reshape,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state],
                                        feed_dict=feed)

            c = pick_top_n(preds, len(CharRNN.vocab))
            samples.append(CharRNN.int_to_vocab[c])

    return ''.join(samples)


# Here, pass in the path to a checkpoint and sample from the network.



lstm_size = 512
# 'Physics', 'Mathematics', 'Computer Science', 'Quantitative Biology', 'Quantitative Finance', 'Statistics', 'Electrical Engineering and Systems Science', 'Economics'
subject=[0,0,0,0,0,0,0,1]
#
checkpoint = tf.train.latest_checkpoint('checkpoints')
samp = sample(checkpoint, 1000, lstm_size, len(CharRNN.vocab), prime="We", subject=subject)
print(samp)


checkpoint = 'checkpoints/i102000_l512.ckpt'
samp = sample(checkpoint, 1000, lstm_size, len(CharRNN.vocab), prime="We", subject=subject)
print(samp)


checkpoint = 'checkpoints/i102000_l512.ckpt'
samp = sample(checkpoint, 1000, lstm_size, len(CharRNN.vocab), prime="In", subject=subject)
print(samp)

# In[24]:

# checkpoint = 'checkpoints/i2000_l512.ckpt'
# samp = sample(checkpoint, 1000, lstm_size, len(CharRNN.vocab), prime="We", subject=[0,0,1,0,0,0,0,0])
# print(samp)