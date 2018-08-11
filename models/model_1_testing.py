import tensorflow as tf
import numpy as np
from models import CharRNN
tf.train.get_checkpoint_state('checkpoints')


# In[18]:

def pick_top_n(preds, vocab_size, top_n=5):
    """
    pick the top n results

    preds:
    vocab_size
    top_n
    """
    p = np.squeeze(preds)
    # set the rest predictions 0
    p[np.argsort(p)[:-top_n]] = 0
    # regularization
    p = p / np.sum(p)
    # pick a random one
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


# In[19]:

def sample(checkpoint, n_samples, lstm_size, vocab_size, prime="The "):
    """
    Sampling new text

    checkpoint
    n_sample: length of the sample
    lstm_size: number of hidden nodes
    vocab_size
    prime: 起始文本
    """
    # convert input word to char list
    samples = [c for c in prime]
    # sampling=True means batch of size=1 x 1
    model = CharRNN(len(CharRNN.vocab), lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Restore session
        new_state = sess.run(model.initial_state)
        saver.restore(sess, checkpoint)
        for c in prime:
            x = np.zeros((1, 1))
            # input single char
            x[0, 0] = CharRNN.vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state],
                                        feed_dict=feed)

        c = pick_top_n(preds, len(CharRNN.vocab))
        # add new predictions to the sampling
        samples.append(CharRNN.int_to_vocab[c])

        # generate new chars till the limit
        for i in range(n_samples):
            x[0, 0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state],
                                        feed_dict=feed)

            c = pick_top_n(preds, len(CharRNN.vocab))
            samples.append(CharRNN.int_to_vocab[c])

    return ''.join(samples)


# Here, pass in the path to a checkpoint and sample from the network.

# In[20]:

tf.train.latest_checkpoint('checkpoints')

lstm_size = 512
# In[26]:

# 选用最终的训练参数作为输入进行文本生成
checkpoint = tf.train.latest_checkpoint('checkpoints')
samp = sample(checkpoint, 2000, lstm_size, len(CharRNN.vocab), prime="We")
print(samp)

# In[22]:

checkpoint = 'checkpoints/i1000_l512.ckpt'
samp = sample(checkpoint, 1000, lstm_size, len(CharRNN.vocab), prime="We")
print(samp)

# In[23]:

checkpoint = 'checkpoints/i1000_l512.ckpt'
samp = sample(checkpoint, 1000, lstm_size, len(CharRNN.vocab), prime="In")
print(samp)

# In[24]:

checkpoint = 'checkpoints/i2000_l512.ckpt'
samp = sample(checkpoint, 1000, lstm_size, len(CharRNN.vocab), prime="We")
print(samp)