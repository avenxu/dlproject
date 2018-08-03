import tensorflow as tf
import numpy as np
from models import CharRNN
# check checkpoints
tf.train.get_checkpoint_state('checkpoints')


# # 5 文本生成
# 现在我们可以基于我们的训练参数进行文本的生成。当我们输入一个字符时，LSTM会预测下一个字符，我们再将新的字符进行输入，这样能不断的循环下去生成本文。
#
# 为了减少噪音，每次的预测值我会选择最可能的前5个进行随机选择，比如输入h，预测结果概率最大的前五个为[o,e,i,u,b]，我们将随机从这五个中挑选一个作为新的字符，让过程加入随机因素会减少一些噪音的生成。

# In[18]:

def pick_top_n(preds, vocab_size, top_n=5):
    """
    从预测结果中选取前top_n个最可能的字符

    preds: 预测结果
    vocab_size
    top_n
    """
    p = np.squeeze(preds)
    # 将除了top_n个预测值的位置都置为0
    p[np.argsort(p)[:-top_n]] = 0
    # 归一化概率
    p = p / np.sum(p)
    # 随机选取一个字符
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


# In[19]:

def sample(checkpoint, n_samples, lstm_size, vocab_size, prime="We"):
    """
    生成新文本

    checkpoint: 某一轮迭代的参数文件
    n_sample: 新闻本的字符长度
    lstm_size: 隐层结点数
    vocab_size
    prime: 起始文本
    """
    # 将输入的单词转换为单个字符组成的list
    samples = [c for c in prime]
    # sampling=True意味着batch的size=1 x 1
    model = CharRNN.CharRNN(len(CharRNN.vocab), lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 加载模型参数，恢复训练
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            # 输入单个字符
            x[0, 0] = CharRNN.vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state],
                                        feed_dict=feed)

        c = pick_top_n(preds, len(CharRNN.vocab))
        # 添加字符到samples中
        samples.append(CharRNN.int_to_vocab[c])

        # 不断生成字符，直到达到指定数目
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

checkpoint = 'checkpoints/i200_l512.ckpt'
samp = sample(checkpoint, 1000, lstm_size, len(CharRNN.vocab), prime="The")
print(samp)

# In[23]:

checkpoint = 'checkpoints/i1000_l512.ckpt'
samp = sample(checkpoint, 1000, lstm_size, len(CharRNN.vocab), prime="We")
print(samp)

# In[24]:

checkpoint = 'checkpoints/i2000_l512.ckpt'
samp = sample(checkpoint, 1000, lstm_size, len(CharRNN.vocab), prime="We")
print(samp)