import numpy as np
import time
from sklearn.model_selection import StratifiedKFold

from simple_classifier import simple_classifier
from multi_label_classifier import multi_label_classifier


def apply_classification(load_file_name="acceptor",
                         results_log_file="../results/results_log",
                         samples_per_file=10000,
                         pre_length=300,
                         post_length=300):

    start = time.time()
    seed = 12
    np.random.seed(seed)

    print("Reading data")
    x_data = np.load(file="../data/x_" + load_file_name + "_" + str(samples_per_file) + "_samples_" + str(
        pre_length) + "_pre_" + str(post_length) + "_post" + ".npy")
    y_data = np.load(file="../data/y_" + load_file_name + "_" + str(samples_per_file) + "_samples.npy")
    print("Finished reading data in {}. x_data.shape {}, y_data.shape {}".format(time.time() - start,
                                                                                 x_data.shape,
                                                                                 y_data.shape))

    # Prepare train and test data
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    cv_scores = []

    # Perform Kfold cross validation
    for train, test in kfold.split(x_data, y_data):
        print("Round: {}".format(len(cv_scores) + 1))

        # Execute model
        with open(file=results_log_file, mode='a') as fh:
            multi_label_classifier(x_data=x_data,
                              y_data=y_data,
                              filehandler=fh,
                              cv_scores=cv_scores,
                              train=train,
                              test=test,
                              pre_length=pre_length,
                              post_length=post_length)



    print("Mean: {}, Std: {}".format(np.mean(cv_scores), np.std(cv_scores)))
    print("File name:", load_file_name)

    with open(file=results_log_file, mode='a') as fh:
        print("Classified {}".format(load_file_name), file=fh)
        print("Mean: {}, Std: {}\n".format(np.mean(cv_scores), np.std(cv_scores)), file=fh)
        print("This took {} seconds.\n".format(time.time() - start), file=fh)
        print("\n-------------------------------------------------------------------------------\n", file=fh)

if __name__ == '__main__':
    test_start = time.time()
    apply_classification(load_file_name="acceptor_data",
                         samples_per_file=20000,
                         pre_length=300,
                         post_length=300)
    print("This took {} seconds".format(time.time()-test_start))