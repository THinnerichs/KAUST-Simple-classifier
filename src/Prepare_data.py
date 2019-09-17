import numpy as np
from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import itertools
import time


def prepare_data(include_acceptor=False, include_donor=False, save_file_name="dataset", samples_per_file=10000):
    """
    This function preprocesses the created fasta file.
    It reads the records as numpy arrays, performs a One Hot Encoding and the saves both x_data and y_data seperately in .npy files.

    :param include_acceptor:
    :param include_donor:
    :param save_file_name:
    :param samples_per_file:
    :return:
    """
    # Initialize classes for later processing
    print("Reading data ...")
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)
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

    # Read data and perform transformation
    for a, b in itertools.product(["negative", "positive"], mode_list):
        # Read data
        file_name = "../data/{}_{}.fa".format(a, b)
        print("Processing", file_name)
        my_time = time.time()
        counter = 0

        for record in SeqIO.parse(file_name, "fasta"):
            loop_record = np.array(record.seq, np.character)
            onehot_encoded = onehot_encoder.fit_transform(loop_record.reshape((len(loop_record), 1)))

            x_dataset.append(onehot_encoded)
            counter += 1

            if counter % 2000 == 0:
                print("Processed records", counter, time.time() - my_time)
                my_time = time.time()
            if counter >= samples_per_file:
                break

        # Prepare y labels
        y_dataset.extend(counter * [a+b])

    # Transform data type of datasets
    y_dataset = np.array(y_dataset)
    y_dataset = label_encoder.fit_transform(y_dataset)
    print("y_dataset shape:", y_dataset.shape)

    x_dataset = np.array(x_dataset, dtype=np.int64)
    print("x_dataset shape:", x_dataset.shape)

    print("Finished reading data")

    # save dataset in numpy readable files
    np.save(file="../data/x_" + save_file_name + ".npy", arr=x_dataset)
    np.save(file="../data/y_" + save_file_name + ".npy", arr=y_dataset)

    print("Data saved in ./data/x_" + save_file_name + ".npy and ../data/y_" + save_file_name + ".npy")

    end = time.time()
    print("This took {} seconds.".format(end-start))


if __name__ == '__main__':
    prepare_data(include_acceptor=True,
                 include_donor=False,
                 save_file_name="donor_data_100000",
                 samples_per_file=100000)
