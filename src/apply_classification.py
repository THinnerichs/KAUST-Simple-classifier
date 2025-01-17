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
                         pre_start=None,
                         pre_end=None,
                         post_start=None,
                         post_end=None,
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
                       'albaradei', 'albaradei_up', 'albaradei_down', 'tac', 'tcc', 'pseKNC']:
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
                                                ("_" + str(pre_length) + "_pre" if pre_length!=0 else "_") +
                                                ("_" + str(post_length) + "_post" if post_length!=0 else "") + \
                                                (str(pre_start) + "_pre_start_" if pre_start!=None else "")+ \
                                                (str(pre_end) + "_pre_end_" if pre_end!=None else "")+ \
                                                (str(post_start) + "_post_start_" if post_start!=None else "")+ \
                                                (str(post_end) + "_post_end"if post_end!=None else "") + \
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
                  pre_start=pre_start,
                  pre_end=pre_end,
                  post_start=post_start,
                  post_end=post_end,
                  pre_length=pre_length,
                  post_length=post_length,
                  load_file_name=load_file_name)

    # Perform Kfold cross validation
    round = 1
    for train, test in kfold.split(x_data_dict[list(x_data_dict.keys())[0]], y_data):
        print("Round: {}".format(round))

        model.round = round

        # np.save(file="../data/" + load_file_name + "_round_" + str(round) + "_train_indizes.npy", arr=np.array(train))
        # np.save(file="../data/" + load_file_name + "_round_" + str(round) + "_test_indizes.npy", arr=np.array(test))

        round += 1

        # Execute model
        if len(applied_models) == 0:
            print("No valid model selected.")
            raise Exception

        # cv_scores['acc']=list(range(round-1))
        if "simple_classifier" in applied_models:
            model.simple_classifier(cv_scores=cv_scores,
                                    train=train,
                                    test=test)
        # cv_scores['acc']=list(range(round-1))
        if "multi_label_classifier" in applied_models:
            model.multi_label_classifier(cv_scores=cv_scores,
                                         train=train,
                                         test=test)
        # cv_scores['acc']=list(range(round-1))
        if "svm" in applied_models:
            model.svm(cv_scores=cv_scores, train=train, test=test)
        # cv_scores['acc']=list(range(round-1))
        if "naive_bayes" in applied_models:
            model.naive_bayes(cv_scores=cv_scores, train=train, test=test)
        # cv_scores['acc']=list(range(round-1))
        if "gradient_boosting" in applied_models:
            model.gradient_boosting(cv_scores=cv_scores,
                                    train=train,
                                    test=test)
        if "gaussian_process" in applied_models:
            model.gaussian_process_classifier(cv_scores=cv_scores,
                                              train=train,
                                              test=test)
        if "knn" in applied_models:
            model.knn_classifier(cv_scores=cv_scores,
                                 train=train,
                                 test=test)
        if "random_forest" in applied_models:
            model.random_forest_classifier(cv_scores=cv_scores,
                                           train=train,
                                           test=test)
        if "ada_boost" in applied_models:
            model.ada_boost_classifier(cv_scores=cv_scores,
                                       train=train,
                                       test=test)
        # cv_scores['acc']=list(range(round-1))
        if "DiProDB_classifier" in applied_models:
            model.simple_classifier_on_DiProDB(cv_scores=cv_scores,
                                               train=train,
                                               test=test,
                                               epochs=2)
        # cv_scores['acc']=list(range(round-1))
        if "DiProDB_full_classifier" in applied_models:
            model.simple_classifier_on_DiProDB_full(cv_scores=cv_scores,
                                                    train=train,
                                                    test=test)
        # cv_scores['acc']=list(range(round-1))
        if "trint_classifier" in applied_models:
            model.simple_classifier_on_trint(cv_scores=cv_scores,
                                             train=train,
                                             test=test,
                                             epochs=2,
                                             batch_size=200)
        # cv_scores['acc']=list(range(round-1))
        if "repDNA_IDkmer_classifier" in applied_models:
            model.simple_classifier_on_repDNA_IDKmer(cv_scores=cv_scores,
                                                     train=train,
                                                     test=test,
                                                     epochs=10)
        # cv_scores['acc']=list(range(round-1))
        if "repDNA_DAC_classifier" in applied_models:
            model.simple_classifier_on_repDNA_DAC(cv_scores=cv_scores,
                                                  train=train,
                                                  test=test,
                                                  epochs=10)
        # cv_scores['acc']=list(range(round-1))
        if "repDNA_DCC_classifier" in applied_models:
            model.simple_classifier_on_repDNA_DCC(cv_scores=cv_scores,
                                                  train=train,
                                                  test=test,
                                                  epochs=10)
        # cv_scores['acc']=list(range(round-1))
        if "repDNA_TAC_classifier" in applied_models:
            model.simple_classifier_on_repDNA_TAC(cv_scores=cv_scores,
                                                  train=train,
                                                  test=test,
                                                  epochs=10)
        # cv_scores['acc']=list(range(round-1))
        if "repDNA_TCC_classifier" in applied_models:
            model.simple_classifier_on_repDNA_TCC(cv_scores=cv_scores,
                                                  train=train,
                                                  test=test,
                                                  epochs=10)
        # cv_scores['acc']=list(range(round-1))
        if "repDNA_pseKNC_classifier" in applied_models:
            model.simple_classifier_on_repDNA_PseKNC(cv_scores=cv_scores,
                                                     train=train,
                                                     test=test,
                                                     epochs=10)
        # cv_scores['acc']=list(range(round-1))
        if "repDNA_PC_PseDNC_classifier" in applied_models:
            model.simple_classifier_on_repDNA_PC_PseDNC(cv_scores=cv_scores,
                                                        train=train,
                                                        test=test,
                                                        epochs=10)
        # cv_scores['acc']=list(range(round-1))
        if "repDNA_PC_PseTNC_classifier" in applied_models:
            model.simple_classifier_on_repDNA_PC_PseTNC(cv_scores=cv_scores,
                                                        train=train,
                                                        test=test,
                                                        epochs=10)
        # cv_scores['acc']=list(range(round-1))
        if "repDNA_SC_PseDNC_classifier" in applied_models:
            model.simple_classifier_on_repDNA_SC_PseDNC(cv_scores=cv_scores,
                                                        train=train,
                                                        test=test,
                                                        epochs=10)
        # cv_scores['acc']=list(range(round-1))
        if "repDNA_SC_PseTNC_classifier" in applied_models:
            model.simple_classifier_on_repDNA_SC_PseTNC(cv_scores=cv_scores,
                                                        train=train,
                                                        test=test,
                                                        epochs=10)
        # cv_scores['acc']=list(range(round-1))
        if "overall_classifier" in applied_models:
            model.overall_classifier(cv_scores=cv_scores,
                                     train=train,
                                     test=test,
                                     epochs=15,
                                     batch_size=200)
        # cv_scores['acc']=list(range(round-1))
        if "overall_classifier_test" in applied_models:
            model.lost_hope_overall_model_test()
        # cv_scores['acc']=list(range(round-1))
        if "Albaradei_classifier" in applied_models:
            model.albaradei_classifier(cv_scores=cv_scores,
                                       train=train,
                                       test=test)
        if "draw_models" in applied_models:
            model.draw_models()
            print("Plotted all models.")
            raise Exception
    

    print("Accuracy:\tMean: {}, Std: {}".format(np.mean(cv_scores['acc']), np.std(cv_scores['acc'])))
    print("Precision:\tMean: {}, Std: {}".format(np.mean(cv_scores['prec']), np.std(cv_scores['prec'])))
    print("Recall:\tMean: {}, Std: {}".format(np.mean(cv_scores['rec']), np.std(cv_scores['rec'])))

    print((str(pre_start) + "_pre_start_" if pre_start != None else "") + \
          (str(pre_end) + "_pre_end_" if pre_end != None else "") + \
          (str(post_start) + "_post_start_" if post_start != None else "") + \
          (str(post_end) + "_post_end" if post_end != None else ""))

    print("Accuracy:\tMean: {}, Std: {}".format(np.mean(cv_scores['acc']), np.std(cv_scores['acc'])), file=filehandler)
    print("Precision:\tMean: {}, Std: {}".format(np.mean(cv_scores['prec']), np.std(cv_scores['prec'])), file=filehandler)
    print("Recall:\tMean: {}, Std: {}".format(np.mean(cv_scores['rec']), np.std(cv_scores['rec'])), file=filehandler)

    print("File name:", load_file_name)

    print("Classified {}".format(load_file_name), file=filehandler)
    print((str(pre_start) + "_pre_start_" if pre_start!=None else "")+ \
          (str(pre_end) + "_pre_end_" if pre_end!=None else "")+ \
          (str(post_start) + "_post_start_" if post_start!=None else "")+ \
          (str(post_end) + "_post_end"if post_end!=None else ""), file=filehandler)
    print("This took {} seconds.\n".format(time.time() - start_time), file=filehandler)
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

    apply_classification(applied_models=['gradient_boosting'],
                         load_file_name="acceptor_data",
                         samples_per_file=100000,
                         start=0,
                         datasets=['simple'],
                         pre_length=300,
                         post_length=300)
    apply_classification(applied_models=['gradient_boosting'],
                         load_file_name="donor_data",
                         samples_per_file=100000,
                         start=0,
                         datasets=['simple'],
                         pre_length=300,
                         post_length=300)

    '''
    apply_classification(applied_models=["simple_classifier",
                                         "multi_label_classifier",
                                         "DiProDB_classifier",
                                         "trint_classifier",
                                         "repDNA_IDkmer_classifier",
                                         "repDNA_DAC_classifier",
                                         # "repDNA_DCC_classifier",
                                         "repDNA_TAC_classifier",
                                         "repDNA_TCC_classifier",
                                         "repDNA_pseKNC_classifier",
                                         "repDNA_PC_PseDNC_classifier",
                                         "repDNA_PC_PseTNC_classifier",
                                         "repDNA_SC_PseDNC_classifier",
                                         "repDNA_SC_PseTNC_classifier",
                                         ],
                         load_file_name="acceptor_data",
                         samples_per_file=100000,
                         datasets=['simple', 'dint', 'trint', 'IDkmer', 'kmer', 'dac', 'tac', 'tcc', 'pseKNC', 'PC_PseDNC', 'PC_PseTNC', 'SC_PseDNC', 'SC_PseTNC'],
                         pre_length=300,
                         post_length=300)

    apply_classification(applied_models=["simple_classifier",
                                         "multi_label_classifier",
                                         "DiProDB_classifier",
                                         "trint_classifier",
                                         "repDNA_IDkmer_classifier",
                                         "repDNA_DAC_classifier",
                                         # "repDNA_DCC_classifier",
                                         "repDNA_TAC_classifier",
                                         "repDNA_TCC_classifier",
                                         "repDNA_pseKNC_classifier",
                                         "repDNA_PC_PseDNC_classifier",
                                         "repDNA_PC_PseTNC_classifier",
                                         "repDNA_SC_PseDNC_classifier",
                                         "repDNA_SC_PseTNC_classifier",
                                         ],
                         load_file_name="donor_data",
                         samples_per_file=100000,
                         datasets=['simple', 'dint', 'trint', 'IDkmer', 'kmer', 'dac', 'tac', 'tcc', 'pseKNC', 'PC_PseDNC', 'PC_PseTNC', 'SC_PseDNC', 'SC_PseTNC'],
                         pre_length=300,
                         post_length=300)
    '''

    '''
    apply_classification(applied_models=["Albaradei_classifier"],
                         load_file_name="acceptor_data",
                         samples_per_file=20000,
                         datasets=['albaradei', 'albaradei_up', 'albaradei_down'])
    '''

    '''
    for batch_size in range(50,151,50):
        for start in [i*batch_size for i in range(0,int(300/batch_size))]:
            for end in [i*batch_size for i in range(0,int(300/batch_size))]:
                if start == 0 and batch_size==50:
                    continue
                apply_classification(applied_models=["simple_classifier"],
                                     load_file_name="acceptor_data",
                                     datasets=['simple'],
                                     samples_per_file=100000,
                                     pre_length=0,
                                     post_length=0,
                                     pre_start=start,
                                     pre_end=start+batch_size-1,
                                     post_start=302+end,
                                     post_end=302+end+batch_size-1)

                apply_classification(applied_models=["simple_classifier"],
                                     load_file_name="donor_data",
                                     datasets=['simple'],
                                     samples_per_file=100000,
                                     pre_length=0,
                                     post_length=0,
                                     pre_start=start,
                                     pre_end=start+batch_size-1,
                                     post_start=302+end,
                                     post_end=302+end+batch_size-1)

                apply_classification(applied_models=["DiProDB_classifier"],
                                     load_file_name="acceptor_data",
                                     datasets=['dint'],
                                     samples_per_file=100000,
                                     pre_length=0,
                                     post_length=0,
                                     pre_start=start,
                                     pre_end=start+batch_size-1,
                                     post_start=302+end,
                                     post_end=302+end+batch_size-1)

                apply_classification(applied_models=["DiProDB_classifier"],
                                     load_file_name="donor_data",
                                     datasets=['dint'],
                                     samples_per_file=100000,
                                     pre_length=0,
                                     post_length=0,
                                     pre_start=start,
                                     pre_end=start+batch_size-1,
                                     post_start=302+end,
                                     post_end=302+end+batch_size-1)

                apply_classification(applied_models=["trint_classifier"],
                                     load_file_name="acceptor_data",
                                     datasets=['trint'],
                                     samples_per_file=100000,
                                     pre_length=0,
                                     post_length=0,
                                     pre_start=start,
                                     pre_end=start+batch_size-1,
                                     post_start=302+end,
                                     post_end=302+end+batch_size-1)

                apply_classification(applied_models=["trint_classifier"],
                                     load_file_name="donor_data",
                                     datasets=['trint'],
                                     samples_per_file=100000,
                                     pre_length=0,
                                     post_length=0,
                                     pre_start=start,
                                     pre_end=start+batch_size-1,
                                     post_start=302+end,
                                     post_end=302+end+batch_size-1)

                apply_classification(applied_models=["gradient_boosting"],
                                     load_file_name="acceptor_data",
                                     datasets=['simple'],
                                     samples_per_file=100000,
                                     pre_length=0,
                                     post_length=0,
                                     pre_start=start,
                                     pre_end=start+batch_size-1,
                                     post_start=302+end,
                                     post_end=302+end+batch_size-1)

                apply_classification(applied_models=["gradient_boosting"],
                                     load_file_name="donor_data",
                                     datasets=['simple'],
                                     samples_per_file=100000,
                                     pre_length=0,
                                     post_length=0,
                                     pre_start=start,
                                     pre_end=start+batch_size-1,
                                     post_start=302+end,
                                     post_end=302+end+batch_size-1)

                apply_classification(applied_models=["random_forest"],
                                     load_file_name="acceptor_data",
                                     datasets=['simple'],
                                     samples_per_file=100000,
                                     pre_length=0,
                                     post_length=0,
                                     pre_start=start,
                                     pre_end=start+batch_size-1,
                                     post_start=302+end,
                                     post_end=302+end+batch_size-1)

                apply_classification(applied_models=["random_forest"],
                                     load_file_name="donor_data",
                                     datasets=['simple'],
                                     samples_per_file=100000,
                                     pre_length=0,
                                     post_length=0,
                                     pre_start=start,
                                     pre_end=start+batch_size-1,
                                     post_start=302+end,
                                     post_end=302+end+batch_size-1)
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
