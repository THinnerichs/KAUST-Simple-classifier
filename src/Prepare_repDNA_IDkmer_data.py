import numpy as np

import time

from sklearn.preprocessing import LabelEncoder

from repDNA import nac

def prepare_data_with_repDNA_IDkmer(include_acceptor=False,
                                    include_donor=False,
                                    save_file_name="dataset",
                                    samples_per_file=10000):

    print("Reading repDNA_IDkmer data ...")


    start = time.time()


    # Prepare selected modes
    mode_list = []
    if include_acceptor:
        mode_list.append("acceptor")
    if include_donor:
        mode_list.append("donor")


    # Read data and perform transformation
    idkmer = nac.IDkmer(k=3, upto=True)
    dir = "../data/"
    x_dataset = []
    y_dataset = []
    for b in mode_list:
        for a in ["negative", "positive"]:
            x_dataset.extend(idkmer.make_idkmer_vec(open(dir + a + "_" + b + ".fa"),
                                                 open(dir + "positive_" + b + ".fa"),
                                                 open(dir + "negative_" + b + ".fa"))[:samples_per_file])

        y_dataset.extend(samples_per_file * [a + b])

    # Transform data type of datasets
    label_encoder = LabelEncoder()

    y_dataset = np.array(y_dataset)
    y_dataset = label_encoder.fit_transform(y_dataset)
    print("y_dataset shape:", y_dataset.shape)

    x_dataset = np.array(x_dataset, dtype=np.float)
    print("x_dataset shape:", x_dataset.shape)

    print("Finished reading data")

    x_filename = "../data/x_IDkmer_" + save_file_name + "_" + str(samples_per_file) + "_samples" + ".npy"
    y_filename = "../data/y_" + save_file_name + "_" + str(samples_per_file) + "_samples.npy"
    # save dataset in numpy readable files
    np.save(file=x_filename, arr=x_dataset)
    np.save(file=y_filename, arr=y_dataset)

    print("Data saved in {} and {}.".format(x_filename, y_filename))

    end = time.time()
    print("This took {} seconds.".format(end - start))

if __name__ == '__main__':
    prepare_data_with_repDNA_IDkmer(include_acceptor=True,
                                    include_donor=False,
                                    save_file_name="acceptor_data",
                                    samples_per_file=20000)

    prepare_data_with_repDNA_IDkmer(include_acceptor=False,
                                    include_donor=True,
                                    save_file_name="donor_data",
                                    samples_per_file=20000)

