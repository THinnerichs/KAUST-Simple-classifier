import numpy as np
import time
from sklearn.model_selection import StratifiedKFold

from Models import *

def apply_classification(load_file_name="acceptor_data",
                         dataset="",
                         results_log_file="../results/results_log",
                         samples_per_file=10000,
                         pre_length=300,
                         post_length=300):

    start = time.time()
    seed = 12
    np.random.seed(seed)

    print("Reading data")
    x_data = np.load(file="../data/x_" + dataset + ("_" if len(dataset)!=0 else "")+ load_file_name + "_" + str(samples_per_file) + "_samples_" + str(
        pre_length) + "_pre_" + str(post_length) + "_post" + ".npy")
    y_data = np.load(file="../data/y_" + load_file_name + "_" + str(samples_per_file) + "_samples.npy")
    print("Finished reading data in {}. x_data.shape {}, y_data.shape {}".format(time.time() - start,
                                                                                 x_data.shape,
                                                                                 y_data.shape))

    # Prepare train and test data
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    cv_scores = []

    filehandler = open(file=results_log_file, mode='a')
    model = Model(x_data=x_data,
                  y_data=y_data,
                  filehandler=filehandler,
                  pre_length=pre_length,
                  post_length=post_length)

    # Perform Kfold cross validation
    for train, test in kfold.split(x_data, y_data):
        print("Round: {}".format(len(cv_scores) + 1))

        # Execute model

        # model.simple_classifier(cv_scores=cv_scores,
        #                         train=train,
        #                         test=test)

        # model.multi_label_classifier(cv_scores=cv_scores,
        #                  train=train,
        #                  test=test)

        # model.svm(cv_scores=cv_scores, train=train, test=test)
        # model.naive_bayes(cv_scores=cv_scores, train=train, test=test)

        # model.gradient_boosting(cv_scores=cv_scores,
        #                         train=train,
        #                         test=test)

        model.simple_classifier_on_DiProDB(cv_scores=cv_scores,
                                           train=train,
                                           test=test,
                                           epochs=10)

    print("Mean: {}, Std: {}".format(np.mean(cv_scores), np.std(cv_scores)))
    print("File name:", load_file_name)

    print("Classified {}".format(load_file_name), file=filehandler)
    print("Mean: {}, Std: {}\n".format(np.mean(cv_scores), np.std(cv_scores)), file=filehandler)
    print("This took {} seconds.\n".format(time.time() - start), file=filehandler)
    print("\n-------------------------------------------------------------------------------\n", file=filehandler)

    print("Loss values: (val_loss, val_acc, acc):", model.loss_val_index)

    # print accuracy progress
    print("\nRESULTS:")

    print("Validation accuracy:")
    model.val_accuracy_values = np.array(model.val_accuracy_values)
    for i in range(model.epochs):
        column = model.val_accuracy_values[:, i]
        print("Round: {},\tMean: {},\tStd: {}".format(i, np.mean(column), np.std(column)))

    print("Train accuracy:")
    model.accuracy_values = np.array(model.accuracy_values)
    for i in range(model.epochs):
        column = model.accuracy_values[:, i]
        print("Round: {},\tMean: {},\tStd: {}".format(i, np.mean(column), np.std(column)))


    filehandler.close()


if __name__ == '__main__':
    test_start = time.time()
    apply_classification(load_file_name="acceptor_data",
                         samples_per_file=20000,
                         dataset="dint",
                         pre_length=300,
                         post_length=300)
    apply_classification(load_file_name="donor_data",
                         samples_per_file=20000,
                         dataset="dint",
                         pre_length=300,
                         post_length=300)

    # apply_classification(samples_per_file=20000)
    print("This took {} seconds".format(time.time()-test_start))