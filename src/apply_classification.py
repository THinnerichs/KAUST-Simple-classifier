import numpy as np
import time
from sklearn.model_selection import StratifiedKFold

from Models import *

def apply_classification(applied_model="simple_classifier",
                         load_file_name="acceptor_data",
                         dataset="",
                         results_log_file="../results/results_log",
                         samples_per_file=10000,
                         pre_length=300,
                         post_length=300):

    start = time.time()
    seed = 12
    np.random.seed(seed)

    print("Reading data")
    x_data = np.load(file="../data/x_" + dataset + ("_" if len(dataset)!=0 else "")+ load_file_name + "_" + str(samples_per_file) + "_samples" + ("_" + str(
        pre_length) + "_pre" if pre_length!=0 else "") + ("_" + str(post_length) + "_post" if post_length!=0 else "") + ".npy")
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
                  post_length=post_length,
                  load_file_name=load_file_name)

    # Perform Kfold cross validation
    for train, test in kfold.split(x_data, y_data):
        print("Round: {}".format(len(cv_scores) + 1))

        # Execute model
        if applied_model == "simple_classifier":
            model.simple_classifier(cv_scores=cv_scores,
                                 train=train,
                                test=test)
        elif applied_model == "multi_label_classifier":
            model.multi_label_classifier(cv_scores=cv_scores,
                                         train=train,
                                         test=test)
        elif applied_model == "svm":
            model.svm(cv_scores=cv_scores, train=train, test=test)
        elif applied_model == "naive_bayes":
            model.naive_bayes(cv_scores=cv_scores, train=train, test=test)
        elif applied_model == "gradient_boosting":
            model.gradient_boosting(cv_scores=cv_scores,
                                    train=train,
                                    test=test)
        elif applied_model == "DiProDB_classifier":
            model.simple_classifier_on_DiProDB(cv_scores=cv_scores,
                                               train=train,
                                               test=test,
                                               epochs=10,
                                               batch_size=50)
        elif applied_model == "repDNA_classifier":
            model.simple_classifier_on_repDNA(cv_scores=cv_scores,
                                              train=train,
                                              test=test,
                                              epochs=10)
        elif applied_model == "repDNA_IDkmer_classifier":
                    model.simple_classifier_on_repDNA_IDKmer(cv_scores=cv_scores,
                                                             train=train,
                                                             test=test,
                                                             epochs=10)

        elif applied_model == "repDNA_DAC_classifier":
                    model.simple_classifier_on_repDNA_DAC(cv_scores=cv_scores,
                                                          train=train,
                                                          test=test,
                                                          epochs=10)
        elif applied_model == "repDNA_DCC_classifier":
                    model.simple_classifier_on_repDNA_DCC(cv_scores=cv_scores,
                                                          train=train,
                                                          test=test,
                                                          epochs=10)
        elif applied_model == "repDNA_PC_PseDNC_classifier":
                    model.simple_classifier_on_repDNA_PC_PseDNC(cv_scores=cv_scores,
                                                                train=train,
                                                                test=test,
                                                                epochs=10)
        elif applied_model == "repDNA_PC_PseTNC_classifier":
                    model.simple_classifier_on_repDNA_PC_PseTNC(cv_scores=cv_scores,
                                                                train=train,
                                                                test=test,
                                                                epochs=10)
        elif applied_model == "repDNA_SC_PseDNC_classifier":
                    model.simple_classifier_on_repDNA_SC_PseDNC(cv_scores=cv_scores,
                                                                train=train,
                                                                test=test,
                                                                epochs=10)
        elif applied_model == "repDNA_SC_PseTNC_classifier":
                    model.simple_classifier_on_repDNA_SC_PseTNC(cv_scores=cv_scores,
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

    '''
    apply_classification(applied_model="repDNA_SC_PseTNC_classifier",
                         load_file_name="acceptor_data",
                         samples_per_file=20000,
                         dataset="SC_PseTNC",
                         pre_length=0,
                         post_length=0)

    apply_classification(applied_model="repDNA_SC_PseTNC_classifier",
                         load_file_name="donor_data",
                         samples_per_file=20000,
                         dataset="SC_PseTNC",
                         pre_length=0,
                         post_length=0)


    apply_classification(applied_model="repDNA_PC_PseTNC_classifier",
                         load_file_name="acceptor_data",
                         samples_per_file=20000,
                         dataset="PC_PseTNC",
                         pre_length=0,
                         post_length=0)

    apply_classification(applied_model="repDNA_PC_PseTNC_classifier",
                         load_file_name="donor_data",
                         samples_per_file=20000,
                         dataset="PC_PseTNC",
                         pre_length=0,
                         post_length=0)

    '''

    apply_classification(applied_model="repDNA_SC_PseDNC_classifier",
                         load_file_name="acceptor_data",
                         samples_per_file=20000,
                         dataset="SC_PseDNC",
                         pre_length=0,
                         post_length=0)

    apply_classification(applied_model="repDNA_SC_PseDNC_classifier",
                         load_file_name="donor_data",
                         samples_per_file=20000,
                         dataset="SC_PseDNC",
                         pre_length=0,
                         post_length=0)

    '''
    apply_classification(applied_model="repDNA_PC_PseDNC_classifier",
                         load_file_name="acceptor_data",
                         samples_per_file=20000,
                         dataset="PC_PseDNC",
                         pre_length=0,
                         post_length=0)

    apply_classification(applied_model="repDNA_PC_PseDNC_classifier",
                         load_file_name="donor_data",
                         samples_per_file=20000,
                         dataset="PC_PseDNC",
                         pre_length=0,
                         post_length=0)

    '''


    # apply_classification(samples_per_file=20000)
    print("This took {} seconds".format(time.time()-test_start))
