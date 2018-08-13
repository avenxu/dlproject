import numpy as np
import keras
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(seq_size):
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    text = list(open("../data/201777abstract.txt", "r", encoding='latin-1').read())

    vocab = sorted(set(text))
    vocab_to_int = {c: i for i, c in enumerate(vocab)}
    int_to_vocab = dict(enumerate(vocab))

    arr = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

    input = []
    initial_input = [0] * seq_size
    input.append(initial_input)
    label = [arr[0]]
    for n in range(len(arr)):
        # inputs
        if(n < len(arr)):
            new_input = input[-1]
            new_input.append(arr[n])
            new_input.pop(0)
            input.append(new_input)
            if n + 1 < len(arr):
                label.append(arr[n + 1])
            else:
                label.append(arr[0])

    # print(x_text)
    # Generate labels
    targets = np.array(label)
    one_hot_targets = np.eye(len(vocab))[targets]
    return np.array(input), one_hot_targets, vocab_to_int, int_to_vocab



def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

def pick_top(preds, vocab_size):

    c = keras.backend.max(preds, axis=1)
    return c

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    text = list(open("../data/201777abstract.txt", "r", encoding='latin-1').read())

    vocab = sorted(set(text))

    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = dict(enumerate(vocab))
    # Mapping from word to index
    vocabulary = {c: i for i, c in enumerate(vocab)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data(seq_size = 100):
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    x, y, vocab_to_int, int_to_vocab = load_data_and_labels(seq_size)

    return x, y, vocab_to_int, int_to_vocab
