import numpy as np
from load_data import load_char_dic
from keras.models import load_model, Model
from keras_contrib.layers import CRF
from keras_contrib.losses.crf_losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from HelpFuncs import f1, precision, recall

char_to_idx, idx_to_char = load_char_dic()


def sentence_to_matrix(sentence):
    '''
    Convert a sentence to a (1,40,8104) matrix so as to be then imported to the
    trained model.
    '''
    i = 0
    sentence_matrix = np.zeros((1, 40, 8104))
    characters = char_to_idx.keys()
    while len(sentence) > i:
        if sentence[i] in characters:
            char_idx = int(char_to_idx[sentence[i]])
            sentence_matrix[0][i][char_idx] = 1
        else:
            sentence_matrix[0][i][int(char_to_idx['<UKN>'])] = 1
        i += 1

    return sentence_matrix


def matrix_to_tags(matrix_pred):
    '''
    Convert an output matrix (1, 40, 3) to a sequence of tags representing the
    predicted tags of the characters in the imported sentence.
    '''
    idx_to_tag = {'1': 'I', '0': 'O', '2': 'B'}
    pred_idxs = matrix_pred[0].argmax(axis=1)
    pred_tags = [idx_to_tag[str(idx)] for idx in pred_idxs]

    return pred_tags


def TCM_NER(sentence):
    '''
    Named Entity Recognition for TCM related words
    '''
    # load trained model
    model = load_model('./benchmark/BLSTM-0-Dense-0-Epochs-20-1550975755.h5')
    # 添加CRF层
    # custom_objects = {'CRF': CRF, 'crf_loss': crf_loss,
    #                   'crf_viterbi_accuracy': crf_viterbi_accuracy}
    # model = load_model('./models/CRF-BLSTM-0-Dense-0-Epochs-20-1551270515.h5',
    #                    custom_objects=custom_objects)
    sentence_matrix = sentence_to_matrix(sentence)
    pred_matrix = model.predict(sentence_matrix)
    pred_tags = matrix_to_tags(pred_matrix)
    return pred_tags


# print('当归是中药')

print('当归是中药:', ''.join(TCM_NER('当归是中药'))[:(len('当归是中药'))])
print('大黄是中药:', ''.join(TCM_NER('大黄是中药'))[:(len('大黄是中药'))])
# print('当归国华侨回到故乡时')
print('当归国华侨回到故乡时:', ''.join(TCM_NER('当归国华侨回到故乡时'))[:(len('当归国华侨回到故乡时'))])
print('对门家的狗叫大黄:', ''.join(TCM_NER('对门家的狗叫大黄'))[:(len('对门家的狗叫大黄'))])

print('麻黄附子细辛汤的药理作用', ''.join(TCM_NER('麻黄附子细辛汤的药理作用'))[:(len('麻黄附子细辛汤的药理作用'))])
