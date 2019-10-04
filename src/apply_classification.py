import numpy as np
import time
from sklearn.model_selection import StratifiedKFold

import multiprocessing as mp

from Models import *

def apply_classification(applied_models=["simple_classifier"],
                         load_file_name="acceptor_data",
                         datasets=None,
                         results_log_file="../results/results_log",
                         samples_per_file=10000,
                         start=0,
                         pre_length=300,
                         post_length=300):

    start_time = time.time()
    seed = 12
    np.random.seed(seed)

    print("Starting to read data:")
    x_data_dict = {}
    for dataset in datasets:
        print("Reading {} data...".format(dataset))
        if dataset in ['kmer', 'IDkmer', 'dac', 'dcc', 'PC_PseDNC', 'PC_PseTNC', 'SC_PseDNC', 'SC_PseTNC',
                       'albaradei', 'albaradei_up', 'albaradei_down']:
            x_data_dict[dataset] = np.load(file="../data/x_" +
                                                dataset + "_" +
                                                load_file_name +
                                                ("_"+str(start) + "_start" if start != 0 else "") +
                                                "_" + str(samples_per_file) +
                                                "_samples" + ".npy")

        else:
            x_data_dict[dataset] = np.load(file="../data/x_" +
                                                (dataset + "_" if dataset!="simple" else "") +
                                                load_file_name + 
                                                ("_"+str(start) + "_start" if start != 0 else "") + 
                                                "_" + str(samples_per_file) + "_samples" +
                                                ("_" + str(pre_length) + "_pre" if pre_length!=0 else "") +
                                                ("_" + str(post_length) + "_post" if post_length!=0 else "") +
                                                ".npy")

    y_data = np.load(file="../data/y_" + load_file_name + "_" + str(samples_per_file) + "_samples.npy")
    print("Finished reading data in {}. y_data.shape {}".format(time.time() - start_time,
                                                                y_data.shape))
    print("Finished reading all data.")

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
    round = 1
    for train, test in kfold.split(x_data_dict[list(x_data_dict.keys())[0]], y_data):
        print("Round: {}".format(round))

        model.round = round

        np.save(file="../data/round_" + str(round) + "_train_indizes.npy", arr=np.array(train))
        np.save(file="../data/round_" + str(round) + "_test_indizes.npy", arr=np.array(test))

        round += 1

        # Execute model
        if len(applied_models) == 0:
            print("No valid model selected.")
            raise Exception

        if "simple_classifier" in applied_models:
            model.simple_classifier(cv_scores=cv_scores,
                                    train=train,
                                    test=test)
        if "multi_label_classifier" in applied_models:
            model.multi_label_classifier(cv_scores=cv_scores,
                                         train=train,
                                         test=test)
        if "svm" in applied_models:
            model.svm(cv_scores=cv_scores, train=train, test=test)
        if "naive_bayes" in applied_models:
            model.naive_bayes(cv_scores=cv_scores, train=train, test=test)
        if "gradient_boosting" in applied_models:
            model.gradient_boosting(cv_scores=cv_scores,
                                    train=train,
                                    test=test)
        if "DiProDB_classifier" in applied_models:
            model.simple_classifier_on_DiProDB(cv_scores=cv_scores,
                                               train=train,
                                               test=test)
        if "trint_classifier" in applied_models:
            model.simple_classifier_on_trint(cv_scores=cv_scores,
                                             train=train,
                                             test=test,
                                             epochs=2,
                                             batch_size=200)
        if "repDNA_classifier" in applied_models:
            model.simple_classifier_on_repDNA(cv_scores=cv_scores,
                                              train=train,
                                              test=test,
                                              epochs=10)
        if "repDNA_IDkmer_classifier" in applied_models:
            model.simple_classifier_on_repDNA_IDKmer(cv_scores=cv_scores,
                                                     train=train,
                                                     test=test,
                                                     epochs=10)
        if "repDNA_DAC_classifier" in applied_models:
            model.simple_classifier_on_repDNA_DAC(cv_scores=cv_scores,
                                                  train=train,
                                                  test=test,
                                                  epochs=10)
        if "repDNA_DCC_classifier" in applied_models:
            model.simple_classifier_on_repDNA_DCC(cv_scores=cv_scores,
                                                  train=train,
                                                  test=test,
                                                  epochs=10)
        if "repDNA_PC_PseDNC_classifier" in applied_models:
            model.simple_classifier_on_repDNA_PC_PseDNC(cv_scores=cv_scores,
                                                        train=train,
                                                        test=test,
                                                        epochs=10)
        if "repDNA_PC_PseTNC_classifier" in applied_models:
            model.simple_classifier_on_repDNA_PC_PseTNC(cv_scores=cv_scores,
                                                        train=train,
                                                        test=test,
                                                        epochs=10)
        if "repDNA_SC_PseDNC_classifier" in applied_models:
            model.simple_classifier_on_repDNA_SC_PseDNC(cv_scores=cv_scores,
                                                        train=train,
                                                        test=test,
                                                        epochs=10)
        if "repDNA_SC_PseTNC_classifier" in applied_models:
            model.simple_classifier_on_repDNA_SC_PseTNC(cv_scores=cv_scores,
                                                        train=train,
                                                        test=test,
                                                        epochs=10)
        if "overall_classifier" in applied_models:
            model.overall_classifier(cv_scores=cv_scores,
                                     train=train,
                                     test=test,
                                     epochs=15,
                                     batch_size=200)
        if "overall_classifier_test" in applied_models:
            model.lost_hope_overall_model_test()
        if "Albaradei_classifier" in applied_models:
            model.albaradei_classifier(cv_scores=cv_scores,
                                       train=train,
                                       test=test)
        if "draw_models" in applied_models:
            model.draw_models()
            print("Plotted all models.")
            raise Exception


    # print("Accuracy:\tMean: {}, Std: {}".format(np.mean(cv_scores['acc']), np.std(cv_scores['acc'])))
    # print("Precision:\tMean: {}, Std: {}".format(np.mean(cv_scores['prec']), np.std(cv_scores['prec'])))
    # print("Recall:\tMean: {}, Std: {}".format(np.mean(cv_scores['rec']), np.std(cv_scores['rec'])))

    # print("Accuracy:\tMean: {}, Std: {}".format(np.mean(cv_scores['acc']), np.std(cv_scores['acc'])), file=filehandler)
    # print("Precision:\tMean: {}, Std: {}".format(np.mean(cv_scores['prec']), np.std(cv_scores['prec'])), file=filehandler)
    # print("Recall:\tMean: {}, Std: {}".format(np.mean(cv_scores['rec']), np.std(cv_scores['rec'])), file=filehandler)

    # print("File name:", load_file_name)

    # print("Classified {}".format(load_file_name), file=filehandler)
    # print("This took {} seconds.\n".format(time.time() - start_time), file=filehandler)
    # print("\n-------------------------------------------------------------------------------\n", file=filehandler)

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
    apply_classification(applied_models=["repDNA_PC_PseDNC_classifier"],
                         load_file_name="acceptor_data",
                         samples_per_file=100000,
                         datasets=["PC_PseDNC"],
                         pre_length=0,
                         post_length=0)
def j2():
    apply_classification(applied_models=["repDNA_PC_PseDNC_classifier"],
                         load_file_name="donor_data",
                         samples_per_file=100000,
                         datasets=["PC_PseDNC"],
                         pre_length=0,
                         post_length=0)
def j3():
    apply_classification(applied_models=["repDNA_PC_PseTNC_classifier"],
                         load_file_name="acceptor_data",
                         samples_per_file=100000,
                         datasets=["PC_PseTNC"],
                         pre_length=0,
                         post_length=0)
def j4():
    apply_classification(applied_models=["repDNA_PC_PseTNC_classifier"],
                         load_file_name="donor_data",
                         samples_per_file=100000,
                         datasets=["PC_PseTNC"],
                         pre_length=0,
                         post_length=0)
def j5():
    apply_classification(applied_models=["repDNA_SC_PseDNC_classifier"],
                         load_file_name="acceptor_data",
                         samples_per_file=100000,
                         datasets=["SC_PseDNC"],
                         pre_length=0,
                         post_length=0)
def j6():
    apply_classification(applied_models=["repDNA_SC_PseDNC_classifier"],
                         load_file_name="donor_data",
                         samples_per_file=100000,
                         datasets=["SC_PseDNC"],
                         pre_length=0,
                         post_length=0)
def j7():
    apply_classification(applied_models=["repDNA_SC_PseTNC_classifier"],
                         load_file_name="acceptor_data",
                         samples_per_file=100000,
                         datasets=["SC_PseTNC"],
                         pre_length=0,
                         post_length=0)
def j8():
    apply_classification(applied_models=["repDNA_SC_PseTNC_classifier"],
                         load_file_name="donor_data",
                         samples_per_file=100000,
                         datasets=["SC_PseTNC"],
                         pre_length=0,
                         post_length=0)
def j9():
    apply_classification(applied_models=["repDNA_DAC_classifier"],
                         load_file_name="acceptor_data",
                         samples_per_file=100000,
                         datasets=["dac"],
                         pre_length=0,
                         post_length=0)
def j10():
    apply_classification(applied_models=["repDNA_DAC_classifier"],
                         load_file_name="donor_data",
                         samples_per_file=100000,
                         datasets=["dac"],
                         pre_length=0,
                         post_length=0)
def j11():
    apply_classification(applied_models=["repDNA_DCC_classifier"],
                         load_file_name="acceptor_data",
                         samples_per_file=100000,
                         datasets=["dcc"],
                         pre_length=0,
                         post_length=0)
def j12():
    apply_classification(applied_models=["repDNA_DCC_classifier"],
                         load_file_name="donor_data",
                         samples_per_file=100000,
                         datasets=["dcc"],
                         pre_length=0,
                         post_length=0)
def j13():
    apply_classification(applied_models=["repDNA_IDkmer_classifier"],
                         load_file_name="acceptor_data",
                         samples_per_file=100000,
                         datasets=["IDkmer"],
                         pre_length=0,
                         post_length=0)
def j14():
    apply_classification(applied_models=["repDNA_IDkmer_classifier"],
                         load_file_name="donor_data",
                         samples_per_file=100000,
                         datasets=["IDkmer"],
                         pre_length=0,
                         post_length=0)



if __name__ == '__main__':
    test_start = time.time()

    '''
    apply_classification(applied_models=["overall_classifier"],
                         load_file_name="acceptor_data",
                         samples_per_file=10000,
                         start=100000,
                         datasets=['simple', 'dint', 'trint', 'kmer', 'IDkmer', 'dac', 'dcc', 'PC_PseDNC', 'PC_PseTNC', 'SC_PseDNC', 'SC_PseTNC'],
                         pre_length=300,
                         post_length=300)

    apply_classification(applied_models=["overall_classifier"],
                         load_file_name="donor_data",
                         samples_per_file=10000,
                         start=100000,
                         datasets=['simple', 'dint', 'trint', 'kmer', 'IDkmer', 'dac', 'dcc', 'PC_PseDNC', 'PC_PseTNC', 'SC_PseDNC', 'SC_PseTNC'],
                         pre_length=300,
                         post_length=300)
    '''
    apply_classification(applied_models=["overall_classifier_test"],
                         load_file_name="acceptor_data",
                         samples_per_file=10000,
                         start=100000,
                         datasets=['simple', 'dint', 'trint', 'kmer', 'IDkmer', 'dac', 'dcc', 'PC_PseDNC', 'PC_PseTNC', 'SC_PseDNC', 'SC_PseTNC'],
                         pre_length=300,
                         post_length=300)

    apply_classification(applied_models=["overall_classifier_test"],
                         load_file_name="donor_data",
                         samples_per_file=10000,
                         start=100000,
                         datasets=['simple', 'dint', 'trint', 'kmer', 'IDkmer', 'dac', 'dcc', 'PC_PseDNC', 'PC_PseTNC', 'SC_PseDNC', 'SC_PseTNC'],
                         pre_length=300,
                         post_length=300)


    '''
    apply_classification(applied_models=["Albaradei_classifier"],
                         load_file_name="acceptor_data",
                         samples_per_file=20000,
                         datasets=['albaradei', 'albaradei_up', 'albaradei_down'])
    '''

    '''
    apply_classification(applied_models=["trint_classifier"],
                         load_file_name="acceptor_data",
                         datasets=['trint'],
                         samples_per_file=100000)

    apply_classification(applied_models=["trint_classifier"],
                         load_file_name="donor_data",
                         datasets=['trint'],
                         samples_per_file=100000)
    '''

    '''
    apply_classification(applied_models=["draw_models"],
                         datasets=['simple'],
                         samples_per_file=20000)

    '''

    '''
    for job in [j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12, j13, j14]:
        p = mp.Process(target=job)
        p.start()
    '''

    # apply_classification(samples_per_file=20000)
    print("This took {} seconds".format(time.time()-test_start))
