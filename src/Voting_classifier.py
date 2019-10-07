import numpy as np

from sklearn.metrics import confusion_matrix


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
            self.data_dict[round]["y_data"] = np.load(file="../data/y_" + self.load_file_name + "_100000_samples.npy")[self.test_indizes]

    def hard_voting(self):
        cv_scores = {'acc':[],
                     'prec':[],
                     'rec':[]}

        for round in range(1,11):
            weights = np.array([1,1,1,1,1,1,1,1,1,1])

            matrix = np.array([])
            for i in range(len(self.datasets)):
                array = self.data_dict[round][self.datasets[i]]
                array = array.reshape((array.shape[0],))
                matrix = np.vstack((matrix, array))

            matrix = (np.transpose(matrix) > 0.5).astype(int)
            y_pred = matrix.dot(weights)
            y_pred /= weights.sum()

            conf_matrix = confusion_matrix(y_true=self.data_dict[round]["y_data"],
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

        print("Accuracy:\tMean: {}, Std: {}".format(np.mean(cv_scores['acc']), np.std(cv_scores['acc'])))
        print("Precision:\tMean: {}, Std: {}".format(np.mean(cv_scores['prec']), np.std(cv_scores['prec'])))
        print("Recall:\tMean: {}, Std: {}".format(np.mean(cv_scores['rec']), np.std(cv_scores['rec'])))


        # print("File name:", load_file_name)

        # print("Classified {}".format(load_file_name), file=filehandler)
        # print("This took {} seconds.\n".format(time.time() - start_time), file=filehandler)
        # print("\n-------------------------------------------------------------------------------\n", file=filehandler)

    def soft_voting(self):
        cv_scores = {'acc':[],
                     'prec':[],
                     'rec':[]}

        for round in range(1,11):
            weights = np.array([1,1,1,1,1,1,1,1,1,1,1])

            matrix = np.array([])
            for i in range(len(self.datasets)):
                matrix = np.vstack((matrix, self.data_dict[round][self.datasets[i]]))

            matrix = np.transpose(matrix)
            y_pred = matrix.dot(weights)
            y_pred /= weights.sum()

            conf_matrix = confusion_matrix(y_true=self.data_dict[round]["y_data"],
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

        print("Accuracy:\tMean: {}, Std: {}".format(np.mean(cv_scores['acc']), np.std(cv_scores['acc'])))
        print("Precision:\tMean: {}, Std: {}".format(np.mean(cv_scores['prec']), np.std(cv_scores['prec'])))
        print("Recall:\tMean: {}, Std: {}".format(np.mean(cv_scores['rec']), np.std(cv_scores['rec'])))


        # print("File name:", load_file_name)

        # print("Classified {}".format(load_file_name), file=filehandler)
        # print("This took {} seconds.\n".format(time.time() - start_time), file=filehandler)
        # print("\n-------------------------------------------------------------------------------\n", file=filehandler)

if __name__ == '__main__':
    democracy = Voting_classifer(load_file_name="acceptor_data")

    # democracy.soft_voting()
    democracy.hard_voting()
