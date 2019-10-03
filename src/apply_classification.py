import numpy as np
import time
from sklearn.model_selection import StratifiedKFold

import multiprocessing as mp

from Models import *

def apply_classification(applied_model="simple_classifier",
                         load_file_name="acceptor_data",
                         datasets=None,
                         results_log_file="../results/results_log",
                         samples_per_file=10000,
                         pre_length=300,
                         post_length=300):

    start = time.time()
    seed = 12
    np.random.seed(seed)

    print("Reading data")
    x_data_dict = {}
    for dataset in datasets:
        print("Reading {} data".format(dataset))
        if dataset in ['kmer', 'IDkmer', 'dac', 'dcc', 'PC_PseDNC', 'PC_PseTNC', 'SC_PseDNC', 'SC_PseTNC',
                       'albaradei', 'albaradei_up', 'albaradei_down']:
            x_data_dict[dataset] = np.load(file="../data/x_" +
                                                dataset + "_" +
                                                load_file_name + "_" + str(samples_per_file) +
                                                "_samples" + ".npy")

        else:
            x_data_dict[dataset] = np.load(file="../data/x_" +
                                                (dataset + "_" if dataset!="simple" else "") +
                                                load_file_name + "_" +
                                                str(samples_per_file) + "_samples" +
                                                ("_" + str(pre_length) + "_pre" if pre_length!=0 else "") +
                                                ("_" + str(post_length) + "_post" if post_length!=0 else "") +
                                                ".npy")

    y_data = np.load(file="../data/y_" + load_file_name + "_" + str(samples_per_file) + "_samples.npy")
    print("Finished reading data in {}. y_data.shape {}".format(time.time() - start,
                                                                y_data.shape))
    for key in list(x_data_dict.keys()):
        print("x_data_shape", key, x_data_dict[key].shape)

    # Prepare train and test data
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    cv_scores = {"acc": [], "prec": [], "rec": []}

    filehandler = open(file=results_log_file, mode='a')
    model = Model(x_data_dict=x_data_dict,
                  y_data=y_data,
                  filehandler=filehandler,
                  pre_length=pre_length,
                  post_length=post_length,
                  load_file_name=load_file_name)

    # Perform Kfold cross validation
    for train, test in kfold.split(x_data_dict[list(x_data_dict.keys())[0]], y_data):
        print("Round: {}".format(len(cv_scores['acc']) + 1))

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
                                               test=test)
        elif applied_model == "trint_classifier":
            model.simple_classifier_on_trint(cv_scores=cv_scores,
                                             train=train,
                                             test=test,
                                             epochs=3)
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
        elif applied_model == "overall_classifier":
            model.samples_per_file = samples_per_file
            model.overall_classifier(cv_scores=cv_scores,
                                     train=train,
                                     test=test,
                                     epochs=15,
                                     batch_size=200)
        elif applied_model == "Albaradei_classifier":
            model.albaradei_classifier(cv_scores=cv_scores,
                                       train=train,
                                       test=test)
        elif applied_model == "draw_models":
            model.draw_models()
            print("Plotted all models.")
            raise Exception
        else:
            print("No valid model selected.")
            raise Exception


    print("Accuracy:\tMean: {}, Std: {}".format(np.mean(cv_scores['acc']), np.std(cv_scores['acc'])))
    print("Precision:\tMean: {}, Std: {}".format(np.mean(cv_scores['prec']), np.std(cv_scores['prec'])))
    print("Recall:\tMean: {}, Std: {}".format(np.mean(cv_scores['rec']), np.std(cv_scores['rec'])))

    print("Accuracy:\tMean: {}, Std: {}".format(np.mean(cv_scores['acc']), np.std(cv_scores['acc'])), file=filehandler)
    print("Precision:\tMean: {}, Std: {}".format(np.mean(cv_scores['prec']), np.std(cv_scores['prec'])), file=filehandler)
    print("Recall:\tMean: {}, Std: {}".format(np.mean(cv_scores['rec']), np.std(cv_scores['rec'])), file=filehandler)

    print("File name:", load_file_name)

    print("Classified {}".format(load_file_name), file=filehandler)
    print("This took {} seconds.\n".format(time.time() - start), file=filehandler)
    print("\n-------------------------------------------------------------------------------\n", file=filehandler)

    '''
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
    '''

    filehandler.close()

def j1():
    apply_classification(applied_model="repDNA_PC_PseDNC_classifier",
                         load_file_name="acceptor_data",
                         samples_per_file=20000,
                         datasets=["PC_PseDNC"],
                         pre_length=0,
                         post_length=0)
def j2():
    apply_classification(applied_model="repDNA_PC_PseDNC_classifier",
                         load_file_name="donor_data",
                         samples_per_file=20000,
                         datasets=["PC_PseDNC"],
                         pre_length=0,
                         post_length=0)
def j3():
    apply_classification(applied_model="repDNA_PC_PseTNC_classifier",
                         load_file_name="acceptor_data",
                         samples_per_file=20000,
                         datasets=["PC_PseTNC"],
                         pre_length=0,
                         post_length=0)
def j4():
    apply_classification(applied_model="repDNA_PC_PseTNC_classifier",
                         load_file_name="donor_data",
                         samples_per_file=20000,
                         datasets=["PC_PseTNC"],
                         pre_length=0,
                         post_length=0)
def j5():
    apply_classification(applied_model="repDNA_SC_PseDNC_classifier",
                         load_file_name="acceptor_data",
                         samples_per_file=20000,
                         datasets=["SC_PseDNC"],
                         pre_length=0,
                         post_length=0)
def j6():
    apply_classification(applied_model="repDNA_SC_PseDNC_classifier",
                         load_file_name="donor_data",
                         samples_per_file=20000,
                         datasets=["SC_PseDNC"],
                         pre_length=0,
                         post_length=0)
def j7():
    apply_classification(applied_model="repDNA_SC_PseTNC_classifier",
                         load_file_name="acceptor_data",
                         samples_per_file=20000,
                         datasets=["SC_PseTNC"],
                         pre_length=0,
                         post_length=0)
def j8():
    apply_classification(applied_model="repDNA_SC_PseTNC_classifier",
                         load_file_name="donor_data",
                         samples_per_file=20000,
                         datasets=["SC_PseTNC"],
                         pre_length=0,
                         post_length=0)


if __name__ == '__main__':
    test_start = time.time()

    apply_classification(applied_model="overall_classifier",
                         load_file_name="acceptor_data",
                         samples_per_file=20000,
                         datasets=['simple', 'dint', 'trint', 'kmer', 'IDkmer', 'dac', 'dcc', 'PC_PseDNC', 'PC_PseTNC', 'SC_PseDNC', 'SC_PseTNC'],
                         pre_length=300,
                         post_length=300)

    apply_classification(applied_model="overall_classifier",
                         load_file_name="donor_data",
                         samples_per_file=20000,
                         datasets=['simple', 'dint', 'trint', 'kmer', 'IDkmer', 'dac', 'dcc', 'PC_PseDNC', 'PC_PseTNC', 'SC_PseDNC', 'SC_PseTNC'],
                         pre_length=300,
                         post_length=300)

    '''
    apply_classification(applied_model="Albaradei_classifier",
                         load_file_name="acceptor_data",
                         samples_per_file=20000,
                         datasets=['albaradei', 'albaradei_up', 'albaradei_down'])
    '''

    apply_classification(applied_model="trint_classifier",
                         load_file_name="acceptor_data",
                         datasets=['trint'],
                         samples_per_file=100000)

    apply_classification(applied_model="trint_classifier",
                         load_file_name="donor_data",
                         datasets=['trint'],
                         samples_per_file=100000)

    '''
    apply_classification(applied_model="draw_models",
                         datasets=['simple'],
                         samples_per_file=20000)

    '''

    '''

    for job in [j1, j2, j3, j4, j5, j6, j7, j8]:
        p = mp.Process(target=job)
        p.start()
    '''

    # apply_classification(samples_per_file=20000)
    print("This took {} seconds".format(time.time()-test_start))
