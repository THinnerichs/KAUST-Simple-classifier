import numpy as np

import itertools

import pickle

import time

from keras.preprocessing.image import img_to_array


def prepare_albaradei_data(include_acceptor=False,
                           include_donor=False,
                           save_file_name="acceptor_data",
                           samples_per_file=10000,
                           pre_length=300,
                           post_length=300):


    def TextToList(fileName):
        dna_list = []
        with open(fileName) as file:
            count = 0
            for line in file:
                li = line.strip()
                if not li.startswith(">"):
                    dna_list.append(line.rstrip("\n"))
                    count += 1
                if count >= samples_per_file:
                    break
        return dna_list

    def split_up_down(dna_list, sig_str, sig_end, begin, end):
        down = []
        up = []
        # short_dna=[]
        for s in range(len(dna_list)):
            up.append(dna_list[s][begin:sig_str])
            down.append(dna_list[s][sig_end:end])
        return up, down

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

    def EncodeSeqToTri_64D(dna_list):
        print("DNA_LIST length:", len(dna_list))
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

            if len(data)>=samples_per_file:
                break

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

    # Prepare selected modes
    mode_list = []
    if include_acceptor:
        mode_list.append("acceptor")
    if include_donor:
        mode_list.append("donor")

    data = []

    # Read data and perform transformation
    for a, b in itertools.product(["negative", "positive"], mode_list):
        data.extend(TextToList("../data/{}_{}.fa".format(a, b)))

    # Processing one hot encoding of data
    print("Reading data...")
    test_data = np.array(EncodeSeqToMono_4D(data), dtype=np.int64)

    print("x_dataset shape:", test_data.shape)

    print("Finished reading data")

    x_filename = "../data/x_albaradei_" + save_file_name + "_" + str(samples_per_file) + "_samples.npy"
    # save dataset in numpy readable files
    np.save(file=x_filename, arr=test_data)

    print("Data saved in {}.".format(x_filename))


    # Processing up and downstream part
    print("Processing downstream data...")
    # split up and down
    sig_str = pre_length
    sig_end = pre_length + 2
    begin = 300 - pre_length
    end = 302 + post_length

    test_up, test_down = split_up_down(data, sig_str, sig_end, begin, end)
    print("LENGTHS:", len(test_up), len(test_down))

    test_images = np.array(EncodeSeqToMono_4D(test_down), dtype=np.int64)
    print("x_dataset shape:", test_images.shape)

    print("Finished reading data")


    x_filename = "../data/x_albaradei_down_" + save_file_name + "_" + str(samples_per_file) + "_samples.npy"
    # save dataset in numpy readable files
    np.save(file=x_filename, arr=test_images)

    print("Data saved in {}.".format(x_filename))


    print("Processing upstream data...")
    # up model
    test_images = np.array(EncodeSeqToTri_64D(test_up), dtype=np.int64)

    print("x_dataset shape:", test_images.shape)

    print("Finished reading data")


    x_filename = "../data/x_albaradei_up_" + save_file_name + "_" + str(samples_per_file) + "_samples.npy"
    # save dataset in numpy readable files
    np.save(file=x_filename, arr=test_images)

    print("Data saved in {}.".format(x_filename))


if __name__ == '__main__':
    start = time.time()
    prepare_albaradei_data(include_acceptor=True,
                           include_donor=False,
                           save_file_name="acceptor_data",
                           samples_per_file=20000)

    prepare_albaradei_data(include_acceptor=False,
                           include_donor=True,
                           save_file_name="donor_data",
                           samples_per_file=20000)

    end = time.time()
    print("This took {} seconds.".format(end-start))

