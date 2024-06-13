import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from load_data import split_dataset
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, LSTM, Reshape, Lambda, Input, Bidirectional, CuDNNLSTM, TimeDistributed
from keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras_contrib.layers import CRF
from keras_contrib.losses.crf_losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from HelpFuncs import f1, precision, recall
import time

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

train_X, dev_X, test_X, train_Y, dev_Y, test_Y = split_dataset()


def train_model_BLSTM_CRF(num_epoch=15, is_train=True):
    n_a = 64
    NAME = 'CRF-BLSTM-%s-Dense-%s-Epochs-%s-%s' % (0,
                                                   0, num_epoch, int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/' + NAME)
    model = Sequential()
    model.add(Bidirectional(CuDNNLSTM(n_a, input_shape=(
        train_X.shape[1], train_X.shape[2]), return_sequences=True, kernel_regularizer=l2(0.001))))

    model.add(TimeDistributed(Dense(n_a, activation='relu')))

    crf = CRF(train_Y.shape[2], learn_mode='marginal',
              sparse_target=False)
    model.add(crf)

    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(optimizer=opt,
                  loss=crf_loss,
                  metrics=[crf_viterbi_accuracy])

    model.fit(train_X, train_Y, batch_size=64, epochs=num_epoch,
              validation_data=(dev_X, dev_Y), callbacks=[tensorboard])

    model.save('./models/'+NAME+'.h5')
    return model


def test_model():
    custom_objects = {'CRF': CRF, 'crf_loss': crf_loss,
                      'crf_viterbi_accuracy': crf_viterbi_accuracy}

    model = load_model('./models/CRF-BLSTM-0-Dense-0-Epochs-20-1551270515.h5',
                       custom_objects=custom_objects)
    test_lost, test_acc = model.evaluate(test_X, test_Y)
    print(test_lost, test_acc)


if __name__ == '__main__':
    # train_model_BLSTM_CRF(num_epoch=20, is_train=True)
    test_model()
