
import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from sys import argv
import time
import warnings
import tensorflow as tf
import os
from tensorflow.python.util import deprecation

from sklearn.metrics import confusion_matrix


def TextToList(fileName):
    dna_list = []
    with open(fileName) as file:

        for line in file:
            li = line.strip()
            if not li.startswith(">"):
                dna_list.append(line.rstrip("\n"))
    file.close()
    return dna_list


# split regons around SS to Up stream and Down strem

def split_up_down(dna_list, sig_str, sig_end, begin, end):
    down = []
    up = []
    # short_dna=[]
    for s in range(len(dna_list)):
        up.append(dna_list[s][begin:sig_str])
        down.append(dna_list[s][sig_end:end])
    return up, down


# encode sequnce to image (4*L)

def EncodeSeqToMono_4D(dna_list):
    data = []
    image = np.zeros((4, len(dna_list[0])))

    alphabet = 'ACGT'
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    for i in range(len(dna_list)):
        image = np.zeros((4, len(dna_list[0])))
        x = dna_list[i]
        integer_encoded = [char_to_int[char] for char in x]

        j = 0
        for value in integer_encoded:
            if (value == 3):
                image[value][j] += 1
            if (value == 2):
                image[value][j] += 0.5

            image[value][j] += 1
            j = j + 1

        data.append(img_to_array(image))

    return data


# encode sequnce to image (64*L)

def EncodeSeqToTri_64D(dna_list):
    seq = dna_list[0]
    n = len(seq)
    profile = {'AAA': [0] * n, 'ACA': [0] * n, 'AGA': [0] * n, 'ATA': [0] * n,
               'CAA': [0] * n, 'CCA': [0] * n, 'CGA': [0] * n, 'CTA': [0] * n,
               'GAA': [0] * n, 'GCA': [0] * n, 'GGA': [0] * n, 'GTA': [0] * n,
               'TAA': [0] * n, 'TCA': [0] * n, 'TGA': [0] * n, 'TTA': [0] * n,

               'AAC': [0] * n, 'ACC': [0] * n, 'AGC': [0] * n, 'ATC': [0] * n,
               'CAC': [0] * n, 'CCC': [0] * n, 'CGC': [0] * n, 'CTC': [0] * n,
               'GAC': [0] * n, 'GCC': [0] * n, 'GGC': [0] * n, 'GTC': [0] * n,
               'TAC': [0] * n, 'TCC': [0] * n, 'TGC': [0] * n, 'TTC': [0] * n,

               'AAG': [0] * n, 'ACG': [0] * n, 'AGG': [0] * n, 'ATG': [0] * n,
               'CAG': [0] * n, 'CCG': [0] * n, 'CGG': [0] * n, 'CTG': [0] * n,
               'GAG': [0] * n, 'GCG': [0] * n, 'GGG': [0] * n, 'GTG': [0] * n,
               'TAG': [0] * n, 'TCG': [0] * n, 'TGG': [0] * n, 'TTG': [0] * n,

               'AAT': [0] * n, 'ACT': [0] * n, 'AGT': [0] * n, 'ATT': [0] * n,
               'CAT': [0] * n, 'CCT': [0] * n, 'CGT': [0] * n, 'CTT': [0] * n,
               'GAT': [0] * n, 'GCT': [0] * n, 'GGT': [0] * n, 'GTT': [0] * n,
               'TAT': [0] * n, 'TCT': [0] * n, 'TGT': [0] * n, 'TTT': [0] * n}

    idx = list(profile.keys())
    # print(idx)
    data = []
    labels = []
    image = np.zeros((64, n))
    for seq in dna_list:
        for i in range(len(seq) - 2):
            tri = seq[i] + seq[i + 1] + seq[i + 2]
            if tri in profile.keys():
                image[idx.index(tri)][i] += 1
                # print(idx.index(tri))

        data.append(img_to_array(image))
        image = np.zeros((64, n))

    return data


# check seqence and make sure it contains ACGT letters only
def RemoveNonAGCT(dna_list):
    chars = set('ACGT')
    dna_listACGT = []
    for s in dna_list:
        flag = 0
        for c in s:
            if c not in chars:
                flag = -1
                print('Data are not ACGT')
        if flag == 0:
            dna_listACGT.append(s)

    return dna_listACGT


# load models
def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


begin = 0
end = 602
sig_str = 300
sig_end = 302

org = "at"

Data1 = TextToList("../data/negative_acceptor.fa") 
Data2 = TextToList("../data/positive_acceptor.fa")
Data = Data1 + Data2

# Data=RemoveNonAGCT(Data)


global_model = load_model('../models/AlbaradeiModels/acc_global_model_' + org)
up_model = load_model('../models/AlbaradeiModels/acc_up_model_' + org)
down_model = load_model('../models/AlbaradeiModels/acc_down_model_' + org)
# print(down_model)
finalmodel = '../models/AlbaradeiModels/acc_splicedeep_' + org + '.pkl'
final_model = load_pickle(finalmodel)

start = time.time()

test_images = EncodeSeqToMono_4D(Data)

test = np.array(test_images, )

prediction = global_model.predict(test)

globalfeatures_t = prediction.tolist()

# split up and down

test_up, test_down = split_up_down(Data, sig_str, sig_end, begin, end)

# up model

test_images = EncodeSeqToMono_4D(test_down)
test = np.array(test_images, )

prediction = up_model.predict(test)

upfeatures_t = prediction.tolist()

# down model


test_images = EncodeSeqToTri_64D(test_up)
test = np.array(test_images, )

prediction = down_model.predict(test)
dwonfeatures_t = prediction.tolist()

# final model
d_t = np.zeros((len(Data), 6))
idx = 0

for idx in range(len(Data)):
    d_t[idx][0] = globalfeatures_t[idx][0]

    d_t[idx][1] = globalfeatures_t[idx][1]

    d_t[idx][2] = upfeatures_t[idx][0]

    d_t[idx][3] = upfeatures_t[idx][1]

    d_t[idx][4] = dwonfeatures_t[idx][0]
    d_t[idx][5] = dwonfeatures_t[idx][1]

pred = final_model.predict(d_t)

print("Data shape:", len(Data1), len(Data2))

conf_matrix = confusion_matrix(y_true=len(Data1) * [1] + len(Data2) * [0],
                               y_pred=pred)
print("Confusion matrix", conf_matrix)

tp = conf_matrix[0, 0]
tn = conf_matrix[1, 1]
fp = conf_matrix[0, 1]
fn = conf_matrix[1, 0]

precision = tp / (tp + fp) * 100
recall = tp/(tp + fn) * 100
accuracy = (tp + tn) / (tp + tn + fp + fn) * 100

print("Precision", precision)
print("Recall", recall)
print("Accuracy", accuracy)
