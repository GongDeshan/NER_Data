from load_data import split_dataset
import tensorflow as tf
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, LSTM, Reshape, Lambda, Input, Bidirectional, CuDNNLSTM
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session
from keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard
from keras import metrics
import time
import numpy as np
from HelpFuncs import f1, precision, recall


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

train_X, dev_X, test_X, train_Y, dev_Y, test_Y = split_dataset()


def show_data_struc():
    print('Shape of X:', train_X.shape)
    print('Shape of Y:', train_Y.shape)
    print('Number of examples:', train_X.shape[0])
    print('Tx (length of sequence):', train_X.shape[1])
    print('total number of unique values:', train_X.shape[2])


def train_model_LSTM(num_lstm=0, num_dense=0, num_epoch=15):
    n_a = 64
    NAME = 'LSTM-%s-Dense-%s-Epochs-%s-%s' % (num_lstm, num_dense, num_epoch, int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/' + NAME)
    model = Sequential()
    model.add(CuDNNLSTM(n_a, input_shape=(
        train_X.shape[1], train_X.shape[2]), return_sequences=True))

    while num_lstm > 0:
        num_lstm -= 1
        model.add(LSTM(n_a, return_sequences=True, kernel_regularizer=l2(0.05)))

    while num_dense > 0:
        num_dense -= 1
        model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.05)))

    model.add(Dense(train_Y.shape[2], activation='softmax', kernel_regularizer=l2(0.05)))

    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_X, train_Y, epochs=num_epoch, validation_data=(
        dev_X, dev_Y), callbacks=[tensorboard])
    model.save('./models/'+NAME+'.h5')


def train_model_BLSTM(num_blstm=0, num_dense=0, num_epoch=15):
    n_a = 64
    NAME = 'BLSTM-%s-Dense-%s-Epochs-%s-%s' % (num_blstm, num_dense, num_epoch, int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/' + NAME)
    model = Sequential()
    model.add(Bidirectional(CuDNNLSTM(n_a, input_shape=(
        train_X.shape[1], train_X.shape[2]), return_sequences=True, kernel_regularizer=l2(0.001))))
    # model.add(Bidirectional(CuDNNLSTM(n_a, input_shape=(
    #     train_X.shape[1], train_X.shape[2]), return_sequences=True)))

    while num_blstm > 0:
        num_blstm -= 1
        model.add(Bidirectional(LSTM(n_a, return_sequences=True, kernel_regularizer=l2(0.01))))

    while num_dense > 0:
        num_dense -= 1
        model.add(Dense(n_a, activation='relu', kernel_regularizer=l2(0.01)))

    model.add(Dense(train_Y.shape[2], activation='softmax'))
    # model.add(Dense(train_Y.shape[2], activation='softmax'))

    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', precision, recall, f1])
    # model.compile(optimizer=opt,
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    model.fit(train_X, train_Y, batch_size=64, epochs=num_epoch,
              validation_data=(dev_X, dev_Y), callbacks=[tensorboard])
    model.save('./models/' + NAME + '.h5')


def test_model():
    # Currently benchmark is the model:BLSTM-0-Dense-2-Epochs-20-1550329084.h5
    # Its lambda values of the l2 regularztion in the relu activation functions are both 0.02
    custom_objects = {'f1': f1, 'precision': precision, 'recall': recall}
    model = load_model('./models/BLSTM-0-Dense-0-Epochs-30-1551532566.h5',
                       custom_objects=custom_objects)
    test_lost, test_acc, test_precision, test_recall, test_f1 = model.evaluate(test_X, test_Y)
    print(test_lost, test_acc, test_precision, test_recall, test_f1)


if __name__ == '__main__':
    show_data_struc()
    # train_model_LSTM(0, 0, 1)
    # train_model_BLSTM(0, 0, 30)
    # test_model()
    # print('End')
