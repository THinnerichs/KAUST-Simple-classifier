import numpy as np

from sklearn.metrics import confusion_matrix

import time

from keras import models
from keras import layers
from keras.callbacks import TensorBoard
from keras import backend
from keras.models import model_from_json, load_model
from keras.utils import plot_model

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
                         'IDkmer',
                         # 'kmer',
                         'dac',
                         'dcc',
                         'PC_PseDNC',
                         'PC_PseTNC',
                         'SC_PseDNC',
                         'SC_PseTNC'
                         ]
        self.data_dict = {}
        for round in range(1,11):
            self.data_dict[round] = {}
            for dataset in self.datasets:
                print("Reading {}...".format(dataset))
                self.data_dict[round][dataset] = np.load(file="../data/" + dataset + "_" + self.load_file_name + "_round_" + str(round) + "_prediction.npy")

            print("Reading y_data...")
            self.train_indizes = np.load(file="../data/" + self.load_file_name + "_round_" + str(round) + "_train_indizes.npy")
            self.test_indizes = np.load(file="../data/" + self.load_file_name + "_round_" + str(round) + "_test_indizes.npy")
            self.data_dict[round]["y_data"] = np.load(file="../data/y_" + self.load_file_name + "_100000_samples.npy")

    def hard_voting(self,
                    input_weights=np.array([])):

        cv_scores = {'acc':[],
                     'prec':[],
                     'rec':[]}

        start_time = time.time()

        weights =  input_weights if input_weights.size else np.array([1,1,1,1,1,1,1,1,1,1])
        for round in range(1,11):

            matrix = np.array([])
            for i in range(len(self.datasets)):
                array = self.data_dict[round][self.datasets[i]]
                array = array.reshape((array.shape[0],))
                matrix = np.vstack((matrix, array)) if matrix.size else array

            matrix = (np.transpose(matrix) > 0.5).astype(int)
            y_pred = matrix.dot(weights)

            y_pred = np.divide(y_pred, weights.sum())

            conf_matrix = confusion_matrix(y_true=self.data_dict[round]["y_data"][self.test_indizes],
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

        print("HARD VOTING RESULTS:")
        print("Accuracy:\tMean: {}, Std: {}".format(np.mean(cv_scores['acc']), np.std(cv_scores['acc'])))
        print("Precision:\tMean: {}, Std: {}".format(np.mean(cv_scores['prec']), np.std(cv_scores['prec'])))
        print("Recall:\tMean: {}, Std: {}".format(np.mean(cv_scores['rec']), np.std(cv_scores['rec'])))

        with open(file=self.results_log_file, mode='a') as filehandler:
            print("HARD VOTING RESULTS:", file=filehandler)
            print("Accuracy:\tMean: {}, Std: {}".format(np.mean(cv_scores['acc']), np.std(cv_scores['acc'])), file=filehandler)
            print("Precision:\tMean: {}, Std: {}".format(np.mean(cv_scores['prec']), np.std(cv_scores['prec'])), file=filehandler)
            print("Recall:\tMean: {}, Std: {}".format(np.mean(cv_scores['rec']), np.std(cv_scores['rec'])), file=filehandler)
            print("Weights:", weights, file=filehandler)

            print("Classified {}".format(self.load_file_name), file=filehandler)
            print("This took {} seconds.\n".format(time.time() - start_time), file=filehandler)
            print("\n-------------------------------------------------------------------------------\n", file=filehandler)

    def soft_voting(self,
                    input_weights=np.array([])):
        cv_scores = {'acc':[],
                     'prec':[],
                     'rec':[]}

        start_time = time.time()

        weights =  input_weights if input_weights.size else np.array([1,1,1,1,1,1,1,1,1,1])
        for round in range(1,11):

            matrix = np.array([])
            for i in range(len(self.datasets)):
                array = self.data_dict[round][self.datasets[i]]
                array = array.reshape((array.shape[0],))
                matrix = np.vstack((matrix, array)) if matrix.size else array

            matrix = np.transpose(matrix)
            y_pred = matrix.dot(weights)

            y_pred = np.divide(y_pred, weights.sum())

            conf_matrix = confusion_matrix(y_true=self.data_dict[round]["y_data"][self.test_indizes],
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

        print("SOFT VOTING RESULTS:")
        print("Accuracy:\tMean: {}, Std: {}".format(np.mean(cv_scores['acc']), np.std(cv_scores['acc'])))
        print("Precision:\tMean: {}, Std: {}".format(np.mean(cv_scores['prec']), np.std(cv_scores['prec'])))
        print("Recall:\tMean: {}, Std: {}".format(np.mean(cv_scores['rec']), np.std(cv_scores['rec'])))

        with open(file=self.results_log_file, mode='a') as filehandler:
            print("SOFT VOTING RESULTS:", file=filehandler)
            print("Accuracy:\tMean: {}, Std: {}".format(np.mean(cv_scores['acc']), np.std(cv_scores['acc'])), file=filehandler)
            print("Precision:\tMean: {}, Std: {}".format(np.mean(cv_scores['prec']), np.std(cv_scores['prec'])), file=filehandler)
            print("Recall:\tMean: {}, Std: {}".format(np.mean(cv_scores['rec']), np.std(cv_scores['rec'])), file=filehandler)
            print("Weights:", weights, file=filehandler)

            print("Classified {}".format(self.load_file_name), file=filehandler)
            print("This took {} seconds.\n".format(time.time() - start_time), file=filehandler)
            print("\n-------------------------------------------------------------------------------\n", file=filehandler)

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
                array = self.data_dict[round][self.datasets[i]]
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
                      y=self.data_dict[round]["y_data"][self.train_indizes],
                      epochs=epochs,
                      batch_size=batch_size,
                      callbacks=[TensorBoard(log_dir='/tmp/classifier')])

            model.summary()

            # Calculate other validation scores
            y_pred = model.predict(matrix[self.test_indizes])
            conf_matrix = confusion_matrix(y_true=self.data_dict[round]["y_data"][self.test_indizes],
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



if __name__ == '__main__':
    democracy = Voting_classifer(load_file_name="acceptor_data")

    '''
    democracy.soft_voting(np.array([3,2,3,1,1,1,1,1,1,1]))
    democracy.hard_voting(np.array([3,2,3,1,1,1,1,1,1,1]))

    democracy.soft_voting(np.array([5,3,5,1,1,1,1,1,1,1]))
    democracy.hard_voting(np.array([5,3,5,1,1,1,1,1,1,1]))

    democracy.soft_voting(np.array([8,5,8,1,1,1,1,1,1,1]))
    democracy.hard_voting(np.array([8,5,8,1,1,1,1,1,1,1]))

    democracy.soft_voting(np.array([2,1,2,0,0,0,0,0,0,0]))
    democracy.hard_voting(np.array([2,1,2,0,0,0,0,0,0,0]))
    '''

    democracy.neural_net(epochs=10,
                         batch_size=500)
