import numpy as np

from Bio import SeqIO

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import itertools
import time


def prepare_data_with_DiProDB(include_acceptor=False,
                              include_donor=False,
                              save_file_name="dataset",
                              samples_per_file=10000,
                              pre_length=300,
                              post_length=300):

    print("Reading DiProDB data ...")

    start = time.time()

    # Initialize datasets
    x_dataset = []
    y_dataset = []

    # Prepare selected modes
    mode_list = []
    if include_acceptor:
        mode_list.append("acceptor")
    if include_donor:
        mode_list.append("donor")


    # Read DiProDB file
    diProDB_data = np.genfromtxt(fname='../data/DiProDB.txt', delimiter='\t', dtype=str)

    diProDB_data = np.delete(diProDB_data, 0, 1)
    diProDB_data = np.delete(diProDB_data, -1, 1)
    print(diProDB_data.shape)

    pruned = np.transpose(diProDB_data[1:, 1:].astype(np.float))

    scaled = StandardScaler().fit_transform(pruned)

    # Decompose matrix
    pca = PCA(n_components=15)

    principal_components = pca.fit_transform(scaled)

    first_row = diProDB_data[0, 1:]

    dinucleotide_dict = {}
    for i in range(16):
        dinucleotide_dict[first_row[i]] = principal_components[i, :]

    print("Finished.")

    print("Reading sequence files ...")
    # Read data and perform transformation
    N = 2
    for a, b in itertools.product(["negative", "positive"], mode_list):
        # Read data
        file_name = "../data/{}_{}.fa".format(a, b)
        print("Processing", file_name)
        my_time = time.time()
        counter = 0

        for record in SeqIO.parse(file_name, "fasta"):
            loop_record = str(record.seq)[300 - pre_length: 301 + post_length + 1]
            encoded = [dinucleotide_dict[loop_record[i:i+N]] for i in range(len(loop_record) -N+1)]

            x_dataset.append(encoded)
            counter += 1

            if counter % 2000 == 0:
                print("Processed records", counter, ", Time:", time.time() - my_time)
                my_time = time.time()
            if counter >= samples_per_file:
                break

        # Prepare y labels
        y_dataset.extend(counter * [a + b])

    # Transform data type of datasets
    label_encoder = LabelEncoder()

    y_dataset = np.array(y_dataset)
    y_dataset = label_encoder.fit_transform(y_dataset)
    print("y_dataset shape:", y_dataset.shape)

    x_dataset = np.array(x_dataset, dtype=np.float)
    print("x_dataset shape:", x_dataset.shape)

    print("Finished reading data")

    x_filename = "../data/x_dint_" + save_file_name + "_" + str(samples_per_file) + "_samples_" + str(
        pre_length) + "_pre_" + str(post_length) + "_post" + ".npy"
    y_filename = "../data/y_" + save_file_name + "_" + str(samples_per_file) + "_samples.npy"
    # save dataset in numpy readable files
    np.save(file=x_filename, arr=x_dataset)
    np.save(file=y_filename, arr=y_dataset)

    print("Data saved in {} and {}.".format(x_filename, y_filename))

    end = time.time()
    print("This took {} seconds.".format(end - start))

if __name__ == '__main__':
    prepare_data_with_DiProDB(include_acceptor=True,
                              include_donor=False,
                              save_file_name="acceptor_data",
                              samples_per_file=20000)

    prepare_data_with_DiProDB(include_acceptor=False,
                              include_donor=True,
                              save_file_name="donor_data",
                              samples_per_file=20000)
