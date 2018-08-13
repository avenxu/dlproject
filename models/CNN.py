from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split
from models.model_2_data_helper import *
from keras import backend as K
K.tensorflow_backend._get_available_gpus()


print('Loading data')
x, y, vocab_to_int, int_to_vocab = load_data(seq_size = 20)

# x.shape -> (10662, 56)
# y.shape -> (10662, 2)
# len(vocab_to_int) -> 18765
# len(int_to_vocab) -> 18765


# X_train.shape -> (8529, 56)
# y_train.shape -> (8529, 2)
# X_test.shape -> (2133, 56)
# y_test.shape -> (2133, 2)


sequence_length = x.shape[1] #100
vocabulary_size = len(int_to_vocab) # 96
embedding_dim = 96
filter_sizes = [3,4,5]
num_filters = 512
drop = 0.5

epochs = 2
batch_size = 30

# this returns a tensor
print("Creating Model...")
inputs = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(drop)(flatten)
output = Dense(units=vocabulary_size, activation='softmax')(dropout)
# this creates a model that includes
model = Model(inputs=inputs, outputs=output)


checkpoint = ModelCheckpoint('weights.{epoch:03d}-{loss:.4f}.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='auto')
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=adam, loss='categorical_crossentropy')
print("Training Model...")
model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint])  # starts training


prediction = ['W']
initial_input = ([0] * (sequence_length - 1))
initial_input.append(vocab_to_int[prediction[0]])
for n in range(100):
    reshaped = np.reshape(initial_input, (1, sequence_length))
    c_one_hot= model.predict(reshaped)
    c = int_to_vocab[np.argmax(c_one_hot)]
    initial_input.append(vocab_to_int[c])
    initial_input.pop(0)
    prediction.append(c)

result = ''.join(prediction)

np.savetxt("../Logs/CNNPrediction.txt", result)
print(result)