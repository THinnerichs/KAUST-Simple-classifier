import numpy as np

import time

from keras import models
from keras import layers
from keras.callbacks import TensorBoard
from keras import backend
from keras.models import model_from_json, load_model
from keras.utils import plot_model

from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

from scipy.optimize import minimize

from xgboost import XGBClassifier

import tensorflow as tf


class Voting_classifer:

    def __init__(self,
                 results_log_file="../results/results_log",
                 load_file_name="acceptor_data"):
        
        self.results_log_file = results_log_file
        self.load_file_name = load_file_name

        self.round = None

        self.x_data = None

        # Find best epoch
        self.loss_val_index = []
        self.accuracy_values = []
        self.val_accuracy_values = []

        self.epochs = None
        self.batch_size = None
        self.datasets = ['simple',
                         'DiProDB',
                         'trint',
                         'multi_label',
                         'gradient_boosting',
                         'random_forest',
                         'IDkmer',
                         # 'kmer',
                         'dac',
                         # 'dcc',
                         'tac',
                         'tcc',
                         'pseKNC',
                         'PC_PseDNC',
                         'PC_PseTNC',
                         'SC_PseDNC',
                         'SC_PseTNC'
                         ]
        self.data_dict = {}
        self.train_indizes = {}
        self.test_indizes = {}
        print("Reading data...")
        for round in range(1,11):
            self.data_dict[round] = {'train': {},
                                     'test': {}}
            for dataset in self.datasets:
                self.data_dict[round]['test'][dataset] = np.load(file="../data/" + dataset + "_" + self.load_file_name + "_round_" + str(round) + "_prediction.npy")
                self.data_dict[round]['train'][dataset] = np.load(file="../data/" + dataset + "_" + self.load_file_name + "_round_" + str(round) + "_train_prediction.npy")

                if dataset == "multi_label":
                    self.data_dict[round]['test'][dataset] = self.data_dict[round]['test'][dataset].argmax(axis=1)
                    self.data_dict[round]['train'][dataset] = self.data_dict[round]['train'][dataset].argmax(axis=1)

            # print("Reading y_data...")
            self.train_indizes[round] = np.load(file="../data/" + self.load_file_name + "_round_" + str(round) + "_train_indizes.npy")
            self.test_indizes[round] = np.load(file="../data/" + self.load_file_name + "_round_" + str(round) + "_test_indizes.npy")
            self.data_dict["y_data"] = np.load(file="../data/y_" + self.load_file_name + "_100000_samples.npy")

    def voting(self,
               input_weights=np.array([]),
               hard=False):

        print("Starting vote...")
        cv_scores = {'acc':[],
                     'prec':[],
                     'rec':[]}

        start_time = time.time()

        weights =  input_weights if input_weights.size else np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
        for round in range(1,11):

            matrix = np.array([])
            for i in range(len(self.datasets)):
                array = self.data_dict[round]['test'][self.datasets[i]]
                array = array.reshape((array.shape[0],))
                matrix = np.vstack((matrix, array)) if matrix.size else array

            matrix = np.transpose(matrix)
            if hard:
                matrix = (matrix > 0.5).astype(int)

            y_pred = matrix.dot(weights)

            y_pred = np.divide(y_pred, weights.sum())

            conf_matrix = confusion_matrix(y_true=self.data_dict["y_data"][self.test_indizes[round]],
                                           y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int))

            tp = conf_matrix[0, 0]
            tn = conf_matrix[1, 1]
            fp = conf_matrix[0, 1]
            fn = conf_matrix[1, 0]

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            cv_scores['acc'].append(accuracy * 100)
            cv_scores['prec'].append(precision * 100)
            cv_scores['rec'].append(recall * 100)

        print(("HARD" if hard else "SOFT") + " VOTING RESULTS:")
        print("Accuracy:\tMean: {}, Std: {}".format(np.mean(cv_scores['acc']), np.std(cv_scores['acc'])))
        print("Precision:\tMean: {}, Std: {}".format(np.mean(cv_scores['prec']), np.std(cv_scores['prec'])))
        print("Recall:\tMean: {}, Std: {}".format(np.mean(cv_scores['rec']), np.std(cv_scores['rec'])))

        with open(file=self.results_log_file, mode='a') as filehandler:
            print(("HARD" if hard else "SOFT") + " VOTING RESULTS:", file=filehandler)
            print("Accuracy:\tMean: {}, Std: {}".format(np.mean(cv_scores['acc']), np.std(cv_scores['acc'])), file=filehandler)
            print("Precision:\tMean: {}, Std: {}".format(np.mean(cv_scores['prec']), np.std(cv_scores['prec'])), file=filehandler)
            print("Recall:\tMean: {}, Std: {}".format(np.mean(cv_scores['rec']), np.std(cv_scores['rec'])), file=filehandler)
            print("Weights:", weights, file=filehandler)

            print("Classified {}".format(self.load_file_name), file=filehandler)
            print("This took {} seconds.\n".format(time.time() - start_time), file=filehandler)
            print("\n-------------------------------------------------------------------------------\n", file=filehandler)

    def objective_fct_vote(self,
                           weights,
                           round,
                           hard=False):


        matrix = np.array([])
        for i in range(len(self.datasets)):
            array = self.data_dict[round]['test'][self.datasets[i]]
            array = array.reshape((array.shape[0],))
            matrix = np.vstack((matrix, array)) if matrix.size else array

        matrix = np.transpose(matrix)
        if hard:
            matrix = (matrix > 0.5).astype(int)

        y_pred = matrix.dot(weights)

        y_pred = (np.divide(y_pred, weights.sum()) > 0.5).astype(int)

        y_true = self.data_dict["y_data"][self.test_indizes[round]]

        return ((y_pred - y_true)**2).sum()

    def apply_vote_minimize(self,
                            hard=False):

        cv_scores = {'acc': [],
                     'prec': [],
                     'rec': [],
                     'weights': []}

        start_time = time.time()
        for round in range(1,11):
            print("Round", round)

            x0 = np.array([0.1] * 15)
            objective_fct = lambda array: self.objective_fct_vote(array, round=round, hard=False)
            res = minimize(objective_fct, x0=x0, method='Powell')
            weights = res.x

            print("Test objective function")
            print(weights)
            print("minimized weights", objective_fct(weights))
            print("My weights", objective_fct(np.array([5,5,5,4,4,3,1,1,1,1,1,1,1,1,1])))

            print("TRAINING PERFORMANCE:")
            train_matrix = np.array([])
            for i in range(len(self.datasets)):
                array = self.data_dict[round]['train'][self.datasets[i]]
                array = array.reshape((array.shape[0],))
                train_matrix = np.vstack((train_matrix, array)) if train_matrix.size else array

            train_matrix = np.transpose(train_matrix)
            y_pred_train = train_matrix.dot(weights)

            conf_matrix = confusion_matrix(y_true=self.data_dict["y_data"][self.train_indizes[round]],
                                           y_pred=(y_pred_train.reshape((len(y_pred_train))) > 0.5).astype(int))

            tp = conf_matrix[0, 0]
            tn = conf_matrix[1, 1]
            fp = conf_matrix[0, 1]
            fn = conf_matrix[1, 0]

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            print("acc:", accuracy)
            print("pre:", precision)
            print("rec:", recall)

            print("TESTING PERFORMANCE:")
            matrix = np.array([])
            for i in range(len(self.datasets)):
                array = self.data_dict[round]['test'][self.datasets[i]]
                array = array.reshape((array.shape[0],))
                matrix = np.vstack((matrix, array)) if matrix.size else array

            matrix = np.transpose(matrix)
            if hard:
                matrix = (matrix > 0.5).astype(int)

            print("MATRIX", matrix)

            y_pred = matrix.dot(weights)

            y_pred = np.divide(y_pred, weights.sum())

            conf_matrix = confusion_matrix(y_true=self.data_dict["y_data"][self.test_indizes[round]],
                                           y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int))

            tp = conf_matrix[0, 0]
            tn = conf_matrix[1, 1]
            fp = conf_matrix[0, 1]
            fn = conf_matrix[1, 0]

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            cv_scores['acc'].append(accuracy * 100)
            cv_scores['prec'].append(precision * 100)
            cv_scores['rec'].append(recall * 100)
            cv_scores['weights'].append(weights)

            print("acc:", accuracy)
            print("pre:", precision)
            print("rec:", recall)
            print("weights:", weights)


        print(("HARD" if hard else "SOFT") + " MINIMIZE VOTING RESULTS:")
        print("Accuracy:\tMean: {}, Std: {}".format(np.mean(cv_scores['acc']), np.std(cv_scores['acc'])))
        print("Precision:\tMean: {}, Std: {}".format(np.mean(cv_scores['prec']), np.std(cv_scores['prec'])))
        print("Recall:\tMean: {}, Std: {}".format(np.mean(cv_scores['rec']), np.std(cv_scores['rec'])))
        for i in range(10):
            print("Weights: Round {}: {}".format(i+1, cv_scores['weights'][i]))

        with open(file=self.results_log_file, mode='a') as filehandler:
            print(("HARD" if hard else "SOFT") + " MINIMIZE VOTING RESULTS:", file=filehandler)
            print("Accuracy:\tMean: {}, Std: {}".format(np.mean(cv_scores['acc']), np.std(cv_scores['acc'])),
                  file=filehandler)
            print("Precision:\tMean: {}, Std: {}".format(np.mean(cv_scores['prec']), np.std(cv_scores['prec'])),
                  file=filehandler)
            print("Recall:\tMean: {}, Std: {}".format(np.mean(cv_scores['rec']), np.std(cv_scores['rec'])),
                  file=filehandler)
            for i in range(10):
                print("Weights: Round {}: {}".format(i+1, cv_scores['weights'][i]), file=filehandler)
            print("Classified {}".format(self.load_file_name), file=filehandler)
            print("This took {} seconds.\n".format(time.time() - start_time), file=filehandler)
            print("\n-------------------------------------------------------------------------------\n",
                  file=filehandler)

    def neural_net(self,
                   hard=False,
                   epochs=5,
                   batch_size=200):

        cv_scores = {'acc': [],
                     'prec': [],
                     'rec': []}

        start_time = time.time()

        # Prepare data
        for round in range(1,11):

            matrix = np.array([])
            for i in range(len(self.datasets)):
                array = self.data_dict[round][self.datasets[i]][self.train_indizes[round]]
                array = array.reshape((array.shape[0],))
                matrix = np.vstack((matrix, array)) if matrix.size else array

            matrix = np.transpose(matrix)

            if hard:
                matrix = (matrix > 0.5).astype(int)

            # defining model
            input_tensor = layers.Input(shape=(None, matrix.shape[1]))
            dense_1 = layers.Dense(4, activation='relu')(input_tensor)
            dense_2 = layers.Dense(2, activation='relu')(dense_1)
            output_tensor = layers.Dense(1, activation='sigmoid')(dense_2)

            model = models.Model(input_tensor, output_tensor)
            '''
            model = Sequential()
            model.add(Flatten())
            model.add(layers.Dense(30, input_shape=(self.pre_length + 2 + self.post_length, 4), activation='relu'))

            model.add(layers.Dropout(0.5))

            model.add(layers.Dense(2, activation='softmax'))
            '''

            # compile model
            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

            # train model
            model.fit(x=matrix[self.train_indizes],
                      y=self.data_dict["y_data"][self.train_indizes[round]],
                      epochs=epochs,
                      batch_size=batch_size,
                      callbacks=[TensorBoard(log_dir='/tmp/classifier')])

            model.summary()

            # Calculate other validation scores
            y_pred = model.predict(matrix[self.test_indizes[round]])
            conf_matrix = confusion_matrix(y_true=self.data_dict["y_data"][self.test_indizes[round]],
                                           y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int))

            tp = conf_matrix[0, 0]
            tn = conf_matrix[1, 1]
            fp = conf_matrix[0, 1]
            fn = conf_matrix[1, 0]

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            cv_scores['acc'].append(accuracy * 100)
            cv_scores['prec'].append(precision * 100)
            cv_scores['rec'].append(recall * 100)

            if len(cv_scores['acc']) == 10:
                with open(file=self.results_log_file, mode='a') as filehandler:
                    print("NEURAL NET " + ("HARD" if hard else "SOFT") + " CLASSIFICATION APPROACH", file=filehandler)
                    print("Data shape: {}".format(self.x_data.shape), file=filehandler)
                    print("Epochs: {}, Batch size: {}".format(epochs, batch_size), file=filehandler)
                    model.summary(print_fn=lambda x: filehandler.write(x + '\n'))
                    print("Confusion matrix:", conf_matrix)

        with open(file=self.results_log_file, mode='a') as filehandler:
            print("SOFT VOTING RESULTS:", file=filehandler)
            print("Accuracy:\tMean: {}, Std: {}".format(np.mean(cv_scores['acc']), np.std(cv_scores['acc'])), file=filehandler)
            print("Precision:\tMean: {}, Std: {}".format(np.mean(cv_scores['prec']), np.std(cv_scores['prec'])), file=filehandler)
            print("Recall:\tMean: {}, Std: {}".format(np.mean(cv_scores['rec']), np.std(cv_scores['rec'])), file=filehandler)

            print("Classified {}".format(self.load_file_name), file=filehandler)
            print("This took {} seconds.\n".format(time.time() - start_time), file=filehandler)
            print("\n-------------------------------------------------------------------------------\n", file=filehandler)

    def sklearn_classifiers(self,
                            hard=False):

        classifiers = [
            # KNeighborsClassifier(3, n_jobs=32),
            # SVC(kernel="linear", C=0.025),
            # SVC(gamma=2, C=1),
            XGBClassifier(max_depth=3,
                          verbosity=1,
                          n_jobs=32,
                          silent=False),
            # GaussianProcessClassifier(1.0 * RBF(1.0),n_jobs=32),
            # DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_jobs=32, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1, max_iter=1000, verbose=True),
            # AdaBoostClassifier(),
            GaussianNB()]

        names = [# "Nearest Neighbors",
                 # "Linear SVM",
                 # "RBF SVM",
                 "XGBoost",
                 # "Gaussian Process",
                 # "Decision Tree",
                 "Random Forest",
                 "Neural Net",
                 # "AdaBoost",
                 "Naive Bayes"]

        for name, clf in zip(names, classifiers):
            cv_scores = {'acc': [],
                         'prec': [],
                         'rec': []}

            start_time = time.time()
            print("Starting {}...".format(name))

            # Prepare data
            for round in range(1, 11):
                print("Round", round)

                matrix = np.array([])
                for i in range(len(self.datasets)):
                    array = self.data_dict[round]['train'][self.datasets[i]]
                    array = array.reshape((array.shape[0],))
                    matrix = np.vstack((matrix, array)) if matrix.size else array

                matrix = np.transpose(matrix)

                if hard:
                    matrix = (matrix > 0.5).astype(int)

                clf.fit(matrix, self.data_dict["y_data"][self.train_indizes[round]])

                # Prepare matrix for prediction
                matrix = np.array([])
                for i in range(len(self.datasets)):
                    array = self.data_dict[round]['test'][self.datasets[i]]
                    array = array.reshape((array.shape[0],))
                    matrix = np.vstack((matrix, array)) if matrix.size else array

                matrix = np.transpose(matrix)
                y_pred = clf.predict(matrix)
                conf_matrix = confusion_matrix(y_true=self.data_dict["y_data"][self.test_indizes[round]],
                                               y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int))

                tp = conf_matrix[0, 0]
                tn = conf_matrix[1, 1]
                fp = conf_matrix[0, 1]
                fn = conf_matrix[1, 0]

                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                accuracy = (tp + tn) / (tp + tn + fp + fn)

                print("acc:", accuracy)
                print("pre:", precision)
                print("rec:", recall)

                cv_scores['acc'].append(accuracy * 100)
                cv_scores['prec'].append(precision * 100)
                cv_scores['rec'].append(recall * 100)

            with open(file=self.results_log_file, mode='a') as filehandler:
                print(name + " RESULTS:", file=filehandler)
                print("Accuracy:\tMean: {}, Std: {}".format(np.mean(cv_scores['acc']), np.std(cv_scores['acc'])),
                      file=filehandler)
                print("Precision:\tMean: {}, Std: {}".format(np.mean(cv_scores['prec']), np.std(cv_scores['prec'])),
                      file=filehandler)
                print("Recall:\tMean: {}, Std: {}".format(np.mean(cv_scores['rec']), np.std(cv_scores['rec'])),
                      file=filehandler)

                print("Classified {}".format(self.load_file_name), file=filehandler)
                print("This took {} seconds.\n".format(time.time() - start_time), file=filehandler)
                print("\n-------------------------------------------------------------------------------\n", file=filehandler)


if __name__ == '__main__':
    democracy = Voting_classifer(load_file_name="acceptor_data")
    # weights = np.array([6.84537089, 4.799788, 7.10828817, - 6.81821823, 8.02828095, 9.27452162, - 3.27671907, 7.43981739, - 20.78048782, - 10.57354025, 1.51687746, 0.72851907, - 1.10868701, 4.87602188, - 1.67576307])
    # democracy.voting(weights)
    # democracy.voting(weights, hard=True)

    # democracy.apply_vote_minimize()
    # democracy.apply_vote_minimize(hard=True)


    # democracy.voting(np.array([5,5,5,0,0,0,0,0,0,0,0,0,0,0,0]))
    # democracy.voting(np.array([5,5,5,0,0,0,0,0,0,0,0,0,0,0,0]), hard=True)

    # democracy.voting(np.array([5,5,0,0,0,0,0,0,0,0,0,0,0,0,0]))
    # democracy.voting(np.array([5,5,0,0,0,0,0,0,0,0,0,0,0,0,0]), hard=True)

    # democracy.voting(np.array([5,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
    # democracy.voting(np.array([5,0,0,0,0,0,0,0,0,0,0,0,0,0,0]), hard=True)

    democracy.sklearn_classifiers()
    democracy.sklearn_classifiers(hard=True)
    


    democracy = Voting_classifer(load_file_name="donor_data")

    # democracy.voting(np.array([5,5,5,0,0,0,0,0,0,0,0,0,0,0,0]))
    # democracy.voting(np.array([5,5,5,0,0,0,0,0,0,0,0,0,0,0,0]), hard=True)

    # democracy.voting(np.array([5,5,0,0,0,0,0,0,0,0,0,0,0,0,0]))
    # democracy.voting(np.array([5,5,0,0,0,0,0,0,0,0,0,0,0,0,0]), hard=True)

    # democracy.voting(np.array([5,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
    # democracy.voting(np.array([5,0,0,0,0,0,0,0,0,0,0,0,0,0,0]), hard=True)

    # democracy.voting(weights)
    # democracy.voting(weights, hard=True)


    # democracy.apply_vote_minimize()
    # democracy.apply_vote_minimize(hard=True)

    democracy.sklearn_classifiers()
    democracy.sklearn_classifiers(hard=True)


    '''
    democracy.voting(np.array([3,2,3,1,1,1,1,1,1,1]))
    democracy.voting(np.array([3,2,3,1,1,1,1,1,1,1]), hard=True)

    democracy.voting(np.array([5,3,5,1,1,1,1,1,1,1]))
    democracy.voting(np.array([5,3,5,1,1,1,1,1,1,1]), hard=True)

    democracy.voting(np.array([8,5,8,1,1,1,1,1,1,1]))
    democracy.voting(np.array([8,5,8,1,1,1,1,1,1,1]), hard=True)

    democracy.voting(np.array([2,1,2,0,0,0,0,0,0,0]))
    democracy.voting(np.array([2,1,2,0,0,0,0,0,0,0]), hard=True)
    '''

    '''
    democracy.neural_net(epochs=10,
                         batch_size=500)
    '''

    '''
    democracy = Voting_classifer(load_file_name="donor_data")
    democracy.voting(np.array([1,1,1,1,1,1,1,1,1,1]))
    democracy.voting(np.array([1,1,1,1,1,1,1,1,1,1]), hard=True)

    democracy.voting(np.array([2,2,2,1,1,1,1,1,1,1]))
    democracy.voting(np.array([2,2,2,1,1,1,1,1,1,1]), hard=True)

    democracy.voting(np.array([3,3,3,1,1,1,1,1,1,1]))
    democracy.voting(np.array([3,3,3,1,1,1,1,1,1,1]), hard=True)

    democracy.voting(np.array([5,5,5,1,1,1,1,1,1,1]))
    democracy.voting(np.array([5,5,5,1,1,1,1,1,1,1]), hard=True)

    democracy.voting(np.array([1,1,1,0,0,0,0,0,0,0]))
    democracy.voting(np.array([1,1,1,0,0,0,0,0,0,0]), hard=True)

    democracy.voting(np.array([3,2,3,1,1,1,1,1,1,1]))
    democracy.voting(np.array([3,2,3,1,1,1,1,1,1,1]), hard=True)

    democracy.voting(np.array([5,3,5,1,1,1,1,1,1,1]))
    democracy.voting(np.array([5,3,5,1,1,1,1,1,1,1]), hard=True)

    democracy.voting(np.array([8,5,8,1,1,1,1,1,1,1]))
    democracy.voting(np.array([8,5,8,1,1,1,1,1,1,1]), hard=True)

    democracy.voting(np.array([2,1,2,0,0,0,0,0,0,0]))
    democracy.voting(np.array([2,1,2,0,0,0,0,0,0,0]), hard=True)

    '''

