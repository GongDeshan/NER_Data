import numpy as np
import random


def load_char_dic():
    # Load the characters in the character dictionary
    char_to_idx = {}
    idx_to_char = {}

    for line in open('../src/onehot/char_onehot.txt', encoding='utf-8'):
        idx, char = line.split()
        char_to_idx[char] = idx
        idx_to_char[idx] = char

    return char_to_idx, idx_to_char


def load_samples():
    # Load samples from a .txt file
    samples = []

    for line in open('../src/samples/samples.txt', encoding='utf-8'):
        samples.append(line.split())

    return samples


def struct_semples():
    # The loaded data are stored in a 3-dimensional matrix struc_sample(a,b,c)
    # a ---- is equal to the number of the characters in the character dictionary
    # b ---- is equal to the number of the samples loaded for the data
    # c ---- is equal to the number of the longes sentence among the samples

    char_to_idx, idx_to_char = load_char_dic()
    samples = load_samples()
    characters = char_to_idx.keys()
    tag_to_idx = {'I': 1, 'O': 0, 'B': 2}

    size_dic = len(char_to_idx)
    size_tag = len(tag_to_idx)
    len_max_sentence = max([len(sample[0]) for sample in samples])
    size_data = len(samples)
    # Initialize the value and the structure of the X matrix. Shape(m,Tx,n_x)
    struc_samples_X = np.zeros((size_data, len_max_sentence, size_dic))
    # Initialize the value and the structure of the Y matrix. Shape(Ty,m,n_y)
    struc_samples_Y = np.zeros((size_data, len_max_sentence, size_tag))

    for i in range(size_data):
        len_sentence = len(samples[i][0])
        for j in range(len_sentence):
            if samples[i][0][j] not in characters:
                idx_char_j = int(char_to_idx['<UKN>'])
            else:
                idx_char_j = int(char_to_idx[samples[i][0][j]])
                tag_char_j = int(tag_to_idx[samples[i][1][j]])
            struc_samples_X[i, j, idx_char_j] = 1
            struc_samples_Y[i, j, tag_char_j] = 1

    return struc_samples_X, struc_samples_Y


def split_dataset(perc_test=0.15, perc_dev=0.15):
    # Split the samples loaded from the data into a training set, a dev set
    # and a test set. The default values of the percentage of the test set and
    # the dev set are both 0.15. The remaining of the samples constitutethe training set.

    struc_samples_X, struc_samples_Y = struct_semples()
    num_samples, _, _ = struc_samples_X.shape

    size_test = int(num_samples*0.15)
    size_dev = int(num_samples*0.15)
    size_train = num_samples - size_test - size_dev

    train_X = struc_samples_X[:size_train, :, :]
    dev_X = struc_samples_X[size_train: size_train+size_dev, :, :]
    test_X = struc_samples_X[-size_test:, :, :]

    train_Y = struc_samples_Y[:size_train, :, :]
    dev_Y = struc_samples_Y[size_train: size_train+size_dev, :, :]
    test_Y = struc_samples_Y[-size_test:, :, :]

    return train_X, dev_X, test_X, train_Y, dev_Y, test_Y


if __name__ == '__main__':
    split_dataset()
    print('End')
    input()
