import numpy as np

from Bio import SeqIO

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import itertools
import time


def prepare_data(include_acceptor=False,
                 include_donor=False,
                 save_file_name="dataset",
                 samples_per_file=10000,
                 start=0,
                 pre_start=0,
                 pre_end=299,
                 post_start=302,
                 post_end=601,
                 pre_length=0,
                 post_length=0):
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
    start_time = time.time()

    # Initialize datasets
    x_dataset = []
    y_dataset = []

    # Prepare selected modes
    mode_list = []
    if include_acceptor:
        mode_list.append("acceptor")
    if include_donor:
        mode_list.append("donor")

    example_seq = ['G', 'A', 'T', 'C']
    encoded_seq = onehot_encoder.fit_transform(np.array(example_seq).reshape((4, 1)))

    nucleotide_dict = {example_seq[i]: encoded_seq[i] for i in range(len(example_seq))}

    # Read data and perform transformation
    for a, b in itertools.product(["negative", "positive"], mode_list):
        # Read data
        file_name = "../data/{}_{}.fa".format(a, b)
        print("Processing", file_name)
        my_time = time.time()
        counter = 0

        for record in SeqIO.parse(file_name, "fasta"):
            if counter < start:
                counter+=1
                continue
            loop_record = str(record.seq)[pre_start:pre_end+1]+str(record.seq)[300:301+1]+str(record.seq)[post_start:post_end+1]
            onehot_encoded = [nucleotide_dict[loop_record[i]] for i in range(len(loop_record))]

            x_dataset.append(onehot_encoded)
            counter += 1

            if counter >= start and counter % 2000 == 0:
                print("Processed records", counter, time.time() - my_time)
                my_time = time.time()
            if counter >= samples_per_file + start:
                break

        # Prepare y labels
        y_dataset.extend(counter * [a+b])


    # Transform data type of datasets
    y_dataset = np.array(y_dataset)
    y_dataset = label_encoder.fit_transform(y_dataset)
    print("y_dataset shape:", y_dataset.shape)

    x_dataset = np.array(x_dataset, dtype=np.int8)
    print("x_dataset shape:", x_dataset.shape)

    print("Finished reading data")

    x_filename = None
    if pre_length == 300 and post_length == 300:
        x_filename = "../data/x_" + save_file_name + ("_"+str(start) + "_start" if start != 0 else "") + "_" + str(samples_per_file) + "_samples_" + str(pre_length) + "_pre_" + str(post_length) + "_post" + ".npy"
    else:
        x_filename = "../data/x_" + save_file_name + ("_"+str(start) + "_start" if start != 0 else "") + "_" + str(samples_per_file) + "_samples_" + \
                     str(pre_start) + "_pre_start_" + \
                     str(pre_end) + "_pre_end_" + \
                     str(post_start) + "_post_start_" + \
                     str(post_end) + "_post_end" + ".npy"

    y_filename = "../data/y_" + save_file_name + ("_"+str(start) + "_start" if start != 0 else "") + "_" + str(samples_per_file) + "_samples.npy"
    # save dataset in numpy readable files
    np.save(file=x_filename, arr=x_dataset)
    np.save(file=y_filename, arr=y_dataset)

    print("Data saved in {} and {}.".format(x_filename,y_filename))

    end = time.time()
    print("This took {} seconds.".format(end-start_time))


if __name__ == '__main__':
    '''
    prepare_data(include_acceptor=True,
                 include_donor=False,
                 save_file_name="acceptor_data",
                 samples_per_file=20000)

    prepare_data(include_acceptor=False,
                 include_donor=True,
                 save_file_name="donor_data",
                 samples_per_file=20000)
    '''

    for start in [i*50 for i in range(0,5)]:
        for end in [i*50 for i in range(0,5)]:
            prepare_data(include_acceptor=True,
                         include_donor=False,
                         save_file_name="acceptor_data",
                         samples_per_file=100000,
                         pre_start=start,
                         pre_end=start+49,
                         post_start=302+end,
                         post_end=302+end+49)

            prepare_data(include_acceptor=False,
                         include_donor=True,
                         save_file_name="donor_data",
                         samples_per_file=100000,
                         pre_start=start,
                         pre_end=start+49,
                         post_start=302+end,
                         post_end=302+end+49)


    '''
    prepare_data(include_acceptor=True,
                 include_donor=False,
                 save_file_name="acceptor_data",
                 start=100000,
                 samples_per_file=10000)

    prepare_data(include_acceptor=False,
                 include_donor=True,
                 save_file_name="donor_data",
                 start=100000,
                 samples_per_file=10000)
    '''
