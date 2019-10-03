import numpy as np

import time

from sklearn.preprocessing import LabelEncoder

from repDNA import nac

def prepare_data_with_repDNA_IDkmer(include_acceptor=False,
                                    include_donor=False,
                                    save_file_name="dataset",
                                    samples_per_file=10000,
                                    start=0):

    print("Reading repDNA_IDkmer data ...")


    start_time = time.time()


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
    for b in mode_list:
        for a in ["negative", "positive"]:
            x_dataset.extend(idkmer.make_idkmer_vec(open(dir + a + "_" + b + ".fa"),
                                                 open(dir + "positive_" + b + ".fa"),
                                                 open(dir + "negative_" + b + ".fa"))[start:start+samples_per_file])

    x_dataset = np.array(x_dataset, dtype=np.float)
    print("x_dataset shape:", x_dataset.shape)

    print("Finished reading data")

    x_filename = "../data/x_IDkmer_" + save_file_name + (str(start) + "_start" if start != 0 else "") + "_" + str(samples_per_file) + "_samples" + ".npy"
    # save dataset in numpy readable files
    np.save(file=x_filename, arr=x_dataset)

    print("Data saved in {}.".format(x_filename))

    end = time.time()
    print("This took {} seconds.".format(end - start))

if __name__ == '__main__':
    prepare_data_with_repDNA_IDkmer(include_acceptor=True,
                                    include_donor=False,
                                    save_file_name="acceptor_data",
                                    samples_per_file=10000,
                                    start=100000)

    prepare_data_with_repDNA_IDkmer(include_acceptor=False,
                                    include_donor=True,
                                    save_file_name="donor_data",
                                    samples_per_file=10000,
                                    start=100000)

