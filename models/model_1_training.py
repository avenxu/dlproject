import tensorflow as tf
import time
from models import CharRNN


# Hyperparameters
batch_size = 10
num_steps = 100
lstm_size = 512
num_layers = 2
learning_rate = 0.001
keep_prob = 0.5
feature_size = 8

epochs = 5000

save_every_n = 500

model = CharRNN.CharRNN(len(CharRNN.vocab), batch_size=batch_size, num_steps=num_steps, lstm_size=lstm_size, num_layers=num_layers,
                        learning_rate=learning_rate, feature_size=feature_size)

saver = tf.train.Saver(max_to_keep=100)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    counter = 0
    for e in range(epochs):
        new_state = sess.run(model.initial_state)
        loss = 0
        for x, y in CharRNN.get_batches(CharRNN.merged_data_matrix, batch_size, num_steps, feature_size):
            counter += 1
            start = time.time()
            feed = {model.inputs: x[:,:,0],
                    model.targets: y,
                    model.subject: x[:,:,1:],
                    model.keep_prob: keep_prob,
                    model.initial_state: new_state}
            batch_loss, new_state, _ = sess.run([model.loss, model.final_state, model.optimizer], feed_dict=feed)

            end=time.time()

            if counter % 500 == 0:
                print('Epoch: {}/{}'.format(e + 1, epochs),
                      'Training Steps: {}...'.format(counter),
                      'Training loss: {}...'.format(batch_loss),
                      '{:.4f} sec/batch'.format((end-start)))

            if (counter % save_every_n == 0):
                saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))

    saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))

