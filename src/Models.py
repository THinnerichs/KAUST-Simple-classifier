import numpy as np

from keras import models
from keras import layers
from keras.callbacks import TensorBoard
from keras import backend
from keras.models import model_from_json
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier


class Model:

    def __init__(self,
                 x_data, y_data,
                 filehandler,
                 pre_length=300,
                 post_length=300,
                 load_file_name="acceptor_data"):
        self.x_data = x_data
        self.y_data = y_data
        self.filehandler = filehandler
        self.pre_length = pre_length
        self.post_length = post_length
        self.load_file_name = load_file_name
        self.samples_per_file = None

        # Find best epoch
        self.loss_val_index = []
        self.accuracy_values = []
        self.val_accuracy_values = []

        self.epochs = None
        self.batch_size = None

        config = tf.ConfigProto(device_count={'GPU': 0, 'CPU': 16})
        sess = tf.Session(config=config)
        backend.set_session(sess)

    def normalize_labels(self):
        return self.x_data.argmax(axis=2)*2/3 - 1

    def simple_classifier(self,
                          cv_scores,
                          train,
                          test,
                          epochs=5,
                          batch_size=200):

        self.epochs = epochs
        self.batch_size = batch_size

        self.x_data = self.x_data.reshape((self.x_data.shape[0], self.x_data.shape[1], self.x_data.shape[2], 1))

        # defining model
        input_tensor = layers.Input(shape=(self.pre_length + 2 + self.post_length, 4, 1))
        convolutional_1 = layers.Conv2D(32, kernel_size=(2, 4), input_shape=(602, 4, 1))(input_tensor)
        max_pool_1 = layers.MaxPooling2D((2, 1))(convolutional_1)
        convolutional_2 = layers.Conv2D(32, kernel_size=(3, 4), input_shape=(602, 4, 1))(input_tensor)
        max_pool_2 = layers.MaxPooling2D((2, 1))(convolutional_2)
        convolutional_3 = layers.Conv2D(32, kernel_size=(4, 4), input_shape=(602, 4, 1))(input_tensor)
        max_pool_3 = layers.MaxPooling2D((2, 1))(convolutional_3)
        convolutional_4 = layers.Conv2D(32, kernel_size=(5, 4), input_shape=(602, 4, 1))(input_tensor)
        max_pool_4 = layers.MaxPooling2D((2, 1))(convolutional_4)
        convolutional_5 = layers.Conv2D(32, kernel_size=(6, 4), input_shape=(602, 4, 1))(input_tensor)
        max_pool_5 = layers.MaxPooling2D((2, 1))(convolutional_5)
        convolutional_6 = layers.Conv2D(32, kernel_size=(7, 4), input_shape=(602, 4, 1))(input_tensor)
        max_pool_6 = layers.MaxPooling2D((2, 1))(convolutional_6)

        merge_1 = layers.Concatenate(axis=1)([max_pool_1, max_pool_2, max_pool_3, max_pool_4, max_pool_5, max_pool_6])

        flatten = layers.Flatten()(merge_1)
        dense_1 = layers.Dense(128, activation='relu')(flatten)
        dropout_1 = layers.Dropout(0.5)(dense_1)
        dense_2 = layers.Dense(128, activation='relu')(dropout_1)
        dropout_2 = layers.Dropout(0.5)(dense_2)
        output_tensor = layers.Dense(1, activation='sigmoid')(dropout_2)

        model = models.Model(input_tensor, output_tensor)

        # compile mode
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # train mo2el
        history = model.fit(x=self.x_data[train],
                            y=self.y_data[train],
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(self.x_data[test], self.y_data[test]),
                            callbacks=[TensorBoard(log_dir='/tmp/classifier')])

        model.summary()

        self.loss_val_index.append((np.array(history.history["val_loss"]).argmin(),
                                    np.array(history.history["val_acc"]).argmax(),
                                    np.array(history.history["acc"]).argmax()))
        self.val_accuracy_values.append(history.history['val_acc'])
        self.accuracy_values.append(history.history['acc'])

        # evaluate the model
        scores = model.evaluate(self.x_data[test], self.y_data[test], verbose=0)

        print("\n--------------------------------------------------")
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        print("--------------------------------------------------\n")
        cv_scores.append(scores[1] * 100)

        if len(cv_scores) == 10:
            print("BINARY CLASSIFICATION APPROACH", file=self.filehandler)
            print("Data shape: {}".format(self.x_data.shape), file=self.filehandler)
            print("Epochs: {}, Batch size: {}".format(epochs, batch_size), file=self.filehandler)
            model.summary(print_fn=lambda x: self.filehandler.write(x + '\n'))

            # print confusion matrix
            y_pred = model.predict(self.x_data[test])
            print("Confusion matrix:",
                  confusion_matrix(y_true=self.y_data[test],
                                   y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int)),
                  file=self.filehandler)
            print("Confusion matrix:",
                  confusion_matrix(y_true=self.y_data[test],
                                   y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int)))

            # Calculate other validation scores
            conf_matrix = confusion_matrix(y_true=self.y_data[test],
                                           y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int))

            tp = conf_matrix[0, 0]
            tn = conf_matrix[1, 1]
            fp = conf_matrix[0, 1]
            fn = conf_matrix[1, 0]

            precision = tp / (tp + fp) * 100
            recall = tp/(tp + fn) * 100

            print("Recall:", recall, file=self.filehandler)
            print("Precision:", precision, file=self.filehandler)

            print("------------------------------------------------\n")

            # serialize model to JSON
            model_json = model.to_json()
            with open("../models/simple_" + self.load_file_name + "_model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("../models/simple_" + self.load_file_name + "_model.h5")
            print("Saved simple convolutional model to disk.")

    def multi_label_classifier(self,
                               cv_scores,
                               train,
                               test,
                               epochs=10,
                               batch_size=500):

        self.epochs = epochs
        self.batch_size = batch_size
        
        onehot_encoder = OneHotEncoder(sparse=False)

        # prepare One Hot Encoding after kfold
        y_train = onehot_encoder.fit_transform(self.y_data[train].reshape((len(self.y_data[train]), 1)))
        y_test = onehot_encoder.fit_transform(self.y_data[test].reshape((len(self.y_data[test]), 1)))

        # defining model
        input_tensor = layers.Input(shape=(self.pre_length + 2 + self.post_length, 4))
        flatten = layers.Flatten()(input_tensor)
        dense_1 = layers.Dense(30, activation='relu')(flatten)
        dropout_1 = layers.Dropout(0.5)(dense_1)
        output_tensor = layers.Dense(2, activation='softmax')(dropout_1)

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
        model.fit(x=self.x_data[train],
                  y=y_train,
                  epochs=epochs,
                  batch_size=batch_size,
                  callbacks=[TensorBoard(log_dir='/tmp/classifier')])

        model.summary()

        # evaluate the model
        scores = model.evaluate(self.x_data[test], y_test, verbose=0)

        print("\n--------------------------------------------------")
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        print("--------------------------------------------------\n")
        cv_scores.append(scores[1] * 100)

        if len(cv_scores) == 10:
            print("MULTI LABEL APPROACH", file=self.filehandler)
            print("Data shape: {}".format(self.x_data.shape), file=self.filehandler)
            print("Epochs: {}, Batch size: {}".format(epochs, batch_size), file=self.filehandler)
            model.summary(print_fn=lambda x: self.filehandler.write(x + '\n'))

            # print confusion matrix
            y_pred = model.predict(self.x_data[test])
            print("Confusion matrix:", confusion_matrix(y_true=y_test.argmax(axis=1), y_pred=y_pred.argmax(axis=1)),
                  file=self.filehandler)
            print("Confusion matrix:", confusion_matrix(y_true=y_test.argmax(axis=1), y_pred=y_pred.argmax(axis=1)))
            print("-----------------------------------------------------\n")

    def svm(self,
            cv_scores,
            train,
            test):
        clf = svm.SVC(gamma='scale', verbose=True)
        clf.fit(self.normalize_labels()[train], self.y_data[train])

        y_pred = clf.predict(self.x_data.argmax(axis=2)[test])

        conf_matrix = confusion_matrix(y_true=self.y_data[test], y_pred=y_pred)

        tp = conf_matrix[0, 0]
        tn = conf_matrix[1, 1]
        fp = conf_matrix[0, 1]
        fn = conf_matrix[1, 0]
        
        accuracy = (tp + tn)/(tp + tn + fp + fn) * 100
        print("SVM accuracy:", accuracy)
        cv_scores.append(accuracy)

        if len(cv_scores) == 10:
            print("SVM APPROACH", file=self.filehandler)
            print("Data shape: {}".format(self.x_data.shape), file=self.filehandler)
            print("Confusion matrix:", conf_matrix, file=self.filehandler)
            print("-----------------------------------------------------\n")

    def naive_bayes(self,
                    cv_scores,
                    train,
                    test):
        gnb = GaussianNB()
        gnb.fit(self.normalize_labels()[train], self.y_data[train])

        y_pred = gnb.predict(self.normalize_labels()[test])

        conf_matrix = confusion_matrix(y_true=self.y_data[test], y_pred=y_pred)

        tp = conf_matrix[0, 0]
        tn = conf_matrix[1, 1]
        fp = conf_matrix[0, 1]
        fn = conf_matrix[1, 0]

        accuracy = (tp + tn)/(tp + tn + fp + fn) * 100
        print("Naive Bayes accuracy:", accuracy)
        cv_scores.append(accuracy)

        if len(cv_scores) == 10:
            print("NAIVE BAYES APPROACH", file=self.filehandler)
            print("Data shape: {}".format(self.x_data.shape), file=self.filehandler)
            print("Confusion matrix:", conf_matrix, file=self.filehandler)
            print("-----------------------------------------------------\n")

    def gradient_boosting(self,
                          cv_scores,
                          train,
                          test):

        model = XGBClassifier()
        model.fit(self.x_data.argmax(axis=2)[train], self.y_data[train], verbose=True)

        print("Model", model)

        y_pred = model.predict(self.x_data.argmax(axis=2)[test])

        conf_matrix = confusion_matrix(y_true=self.y_data[test], y_pred=y_pred)

        tp = conf_matrix[0, 0]
        tn = conf_matrix[1, 1]
        fp = conf_matrix[0, 1]
        fn = conf_matrix[1, 0]

        accuracy = (tp + tn)/(tp + tn + fp + fn) * 100
        print("Gradient boosting accuracy:", accuracy)
        cv_scores.append(accuracy)

        if len(cv_scores) == 10:
            print("GRADIENT BOOSTING APPROACH", file=self.filehandler)
            print("Data shape: {}".format(self.x_data.shape), file=self.filehandler)
            print("Confusion matrix:", conf_matrix, file=self.filehandler)
            print("-----------------------------------------------------\n")

    def simple_classifier_on_DiProDB(self,
                                     cv_scores,
                                     train,
                                     test,
                                     epochs=5,
                                     batch_size=200):
        self.epochs = epochs
        self.batch_size = batch_size

        # defining model
        input_tensor = layers.Input(shape=(self.pre_length + 2 + self.post_length - 1, 15, 1))

        '''
        convolutional_1_1 = layers.Conv2D(16, kernel_size=(2, 15), activation="relu")(input_tensor)
        max_pool_1_1 = layers.MaxPooling2D((2,1))(convolutional_1_1)
        '''

        convolutional_1_2 = layers.Conv2D(32, kernel_size=(3, 15), activation='relu')(input_tensor)
        max_pool_1_2 = layers.MaxPooling2D((2,1))(convolutional_1_2)

        convolutional_1_3 = layers.Conv2D(32, kernel_size=(4, 15), activation='relu')(input_tensor)
        max_pool_1_3 = layers.MaxPooling2D((2, 1))(convolutional_1_3)

        '''
        convolutional_1_4 = layers.Conv2D(32, kernel_size=(5, 15), activation='relu')(input_tensor)
        max_pool_1_4 = layers.MaxPooling2D((2, 1))(convolutional_1_4)

        convolutional_1_5 = layers.Conv2D(32, kernel_size=(6, 15), activation='relu')(input_tensor)
        max_pool_1_5 = layers.MaxPooling2D((2, 1))(convolutional_1_5)

        convolutional_1_6 = layers.Conv2D(32, kernel_size=(7, 15), activation='relu')(input_tensor)
        max_pool_1_6 = layers.MaxPooling2D((2,1))(convolutional_1_6)

        convolutional_1_7 = layers.Conv2D(32, kernel_size=(8, 15), activation='relu')(input_tensor)
        max_pool_1_7 = layers.MaxPooling2D((2,1))(convolutional_1_7)
        '''

        merge_1 = layers.Concatenate(axis=1)([convolutional_1_2, convolutional_1_3])

        flatten = layers.Flatten()(merge_1)
        dense_1 = layers.Dense(128, activation='relu')(flatten)
        dropout_1 = layers.Dropout(0.5)(dense_1)
        dense_2 = layers.Dense(128, activation='relu')(dropout_1)
        dropout_2 = layers.Dropout(0.5)(dense_2)

        output_tensor = layers.Dense(1, activation='sigmoid')(dropout_2)

        model = models.Model(input_tensor, output_tensor)

        # compile model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        self.x_data = self.x_data.reshape((self.x_data.shape[0], self.x_data.shape[1], self.x_data.shape[2], 1))

        # train model
        history = model.fit(x=self.x_data[train],
                            y=self.y_data[train],
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(self.x_data[test], self.y_data[test]),
                            callbacks=[TensorBoard(log_dir='/tmp/classifier')])

        self.loss_val_index.append((np.array(history.history["val_loss"]).argmin(),
                                    np.array(history.history["val_acc"]).argmax(),
                                    np.array(history.history["acc"]).argmax()))
        self.val_accuracy_values.append(history.history['val_acc'])
        self.accuracy_values.append(history.history['acc'])

        model.summary()

        # evaluate the model
        scores = model.evaluate(self.x_data[test], self.y_data[test], verbose=0)

        print("\n--------------------------------------------------")
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        print("--------------------------------------------------\n")
        cv_scores.append(scores[1] * 100)

        if len(cv_scores) == 10:
            print("DiProDB: BINARY CLASSIFICATION APPROACH", file=self.filehandler)
            print("Data shape: {}".format(self.x_data.shape), file=self.filehandler)
            print("Epochs: {}, Batch size: {}".format(epochs, batch_size), file=self.filehandler)
            model.summary(print_fn=lambda x: self.filehandler.write(x + '\n'))

            # print confusion matrix
            y_pred = model.predict(self.x_data[test])
            print("Confusion matrix:",
                  confusion_matrix(y_true=self.y_data[test],
                                   y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int)),
                  file=self.filehandler)
            print("Confusion matrix:",
                  confusion_matrix(y_true=self.y_data[test],
                                   y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int)))

            # Calculate other validation scores
            conf_matrix = confusion_matrix(y_true=self.y_data[test],
                                           y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int))

            tp = conf_matrix[0, 0]
            tn = conf_matrix[1, 1]
            fp = conf_matrix[0, 1]
            fn = conf_matrix[1, 0]

            precision = tp / (tp + fp) * 100
            recall = tp/(tp + fn) * 100

            print("Recall:", recall, file=self.filehandler)
            print("Precision:", precision, file=self.filehandler)

            print("------------------------------------------------\n")

            # serialize model to JSON
            model_json = model.to_json()
            with open("../models/DiProDB_" + self.load_file_name + "_model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("../models/DiProDB_" + self.load_file_name + "_model.h5")
            print("Saved DiProDB convolutional model to disk.")

    def simple_classifier_on_repDNA_Kmer(self,
                                         cv_scores,
                                         train,
                                         test,
                                         epochs=10,
                                         batch_size=500):

        self.epochs = epochs
        self.batch_size = batch_size

        self.x_data = self.x_data.reshape(self.x_data.shape[0], self.x_data.shape[1], 1)

        # defining model
        input_tensor = layers.Input(shape=(20, 1))

        convolutional_1_1 = layers.Conv1D(16, kernel_size=3, activation='relu')(input_tensor)

        convolutional_1_2 = layers.Conv1D(16, kernel_size=4, activation='relu')(input_tensor)

        convolutional_1_3 = layers.Conv1D(16, kernel_size=5, activation='relu')(input_tensor)

        merge_1 = layers.Concatenate(axis=1)([convolutional_1_1, convolutional_1_2, convolutional_1_3])

        flatten = layers.Flatten()(merge_1)
        dense_1 = layers.Dense(64, activation='relu')(flatten)
        dropout_1 = layers.Dropout(0.5)(dense_1)

        output_tensor = layers.Dense(1, activation='sigmoid')(dropout_1)

        model = models.Model(input_tensor, output_tensor)

        # compile model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # train model
        history = model.fit(x=self.x_data[train],
                            y=self.y_data[train],
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(self.x_data[test], self.y_data[test]),
                            callbacks=[TensorBoard(log_dir='/tmp/classifier')])

        self.loss_val_index.append((np.array(history.history["val_loss"]).argmin(),
                                    np.array(history.history["val_acc"]).argmax(),
                                    np.array(history.history["acc"]).argmax()))
        self.val_accuracy_values.append(history.history['val_acc'])
        self.accuracy_values.append(history.history['acc'])

        model.summary()

        # evaluate the model
        scores = model.evaluate(self.x_data[test], self.y_data[test], verbose=0)

        print("\n--------------------------------------------------")
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        print("--------------------------------------------------\n")
        cv_scores.append(scores[1] * 100)

        if len(cv_scores) == 10:
            print("repDNA: KMER BINARY CLASSIFICATION APPROACH", file=self.filehandler)
            print("Data shape: {}".format(self.x_data.shape), file=self.filehandler)
            print("Epochs: {}, Batch size: {}".format(epochs, batch_size), file=self.filehandler)
            model.summary(print_fn=lambda x: self.filehandler.write(x + '\n'))

            # print confusion matrix
            y_pred = model.predict(self.x_data[test])
            print("Confusion matrix:",
                  confusion_matrix(y_true=self.y_data[test],
                                   y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int)),
                  file=self.filehandler)
            print("Confusion matrix:",
                  confusion_matrix(y_true=self.y_data[test],
                                   y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int)))

            # Calculate other validation scores
            conf_matrix = confusion_matrix(y_true=self.y_data[test],
                                           y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int))

            tp = conf_matrix[0, 0]
            tn = conf_matrix[1, 1]
            fp = conf_matrix[0, 1]
            fn = conf_matrix[1, 0]

            precision = tp / (tp + fp) * 100
            recall = tp/(tp + fn) * 100

            print("Recall:", recall, file=self.filehandler)
            print("Precision:", precision, file=self.filehandler)

            print("------------------------------------------------\n")

            # serialize model to JSON
            model_json = model.to_json()
            with open("../models/kmer_" + self.load_file_name + "_model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("../models/kmer_" + self.load_file_name + "_model.h5")
            print("Saved Kmer convolutional model to disk.")

    def simple_classifier_on_repDNA_IDKmer(self,
                                           cv_scores,
                                           train,
                                           test,
                                           epochs=10,
                                           batch_size=500):
        self.epochs = epochs
        self.batch_size = batch_size

        if self.x_data.ndim == 2:
            scaler = StandardScaler().fit(self.x_data[train])
            self.x_data[train] = scaler.transform(self.x_data[train])

            self.x_data[test] = scaler.transform(self.x_data[test])

        self.x_data = self.x_data.reshape(self.x_data.shape[0], self.x_data.shape[1], 1)

        # defining model
        input_tensor = layers.Input(shape=(6, 1))
        convolutional_1_1 = layers.Conv1D(32, kernel_size=4, activation="relu")(input_tensor)

        flatten = layers.Flatten()(convolutional_1_1)
        dense_1 = layers.Dense(64, activation='relu')(flatten)
        dropout_1 = layers.Dropout(0.5)(dense_1)

        output_tensor = layers.Dense(1, activation='sigmoid')(dropout_1)

        model = models.Model(input_tensor, output_tensor)

        # compile model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # train model
        history = model.fit(x=self.x_data[train],
                            y=self.y_data[train],
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(self.x_data[test], self.y_data[test]),
                            callbacks=[TensorBoard(log_dir='/tmp/classifier')])

        self.loss_val_index.append((np.array(history.history["val_loss"]).argmin(),
                                    np.array(history.history["val_acc"]).argmax(),
                                    np.array(history.history["acc"]).argmax()))
        self.val_accuracy_values.append(history.history['val_acc'])
        self.accuracy_values.append(history.history['acc'])

        model.summary()

        # evaluate the model
        scores = model.evaluate(self.x_data[test], self.y_data[test], verbose=0)

        print("\n--------------------------------------------------")
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        print("--------------------------------------------------\n")
        cv_scores.append(scores[1] * 100)

        if len(cv_scores) == 10:
            print("repDNA: IDKMER BINARY CLASSIFICATION APPROACH", file=self.filehandler)
            print("Data shape: {}".format(self.x_data.shape), file=self.filehandler)
            print("Epochs: {}, Batch size: {}".format(epochs, batch_size), file=self.filehandler)
            model.summary(print_fn=lambda x: self.filehandler.write(x + '\n'))

            # print confusion matrix
            y_pred = model.predict(self.x_data[test])
            print("Confusion matrix:",
                  confusion_matrix(y_true=self.y_data[test],
                                   y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int)),
                  file=self.filehandler)
            print("Confusion matrix:",
                  confusion_matrix(y_true=self.y_data[test],
                                   y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int)))

            # Calculate other validation scores
            conf_matrix = confusion_matrix(y_true=self.y_data[test],
                                           y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int))

            tp = conf_matrix[0, 0]
            tn = conf_matrix[1, 1]
            fp = conf_matrix[0, 1]
            fn = conf_matrix[1, 0]

            precision = tp / (tp + fp) * 100
            recall = tp/(tp + fn) * 100

            print("Recall:", recall, file=self.filehandler)
            print("Precision:", precision, file=self.filehandler)

            print("------------------------------------------------\n")

            # serialize model to JSON
            model_json = model.to_json()
            with open("../models/IDkmer_" + self.load_file_name + "_model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("../models/IDkmer_" + self.load_file_name + "_model.h5")
            print("Saved IDkmer convolutional model to disk.")

    def simple_classifier_on_repDNA_DAC(self,
                                        cv_scores,
                                        train,
                                        test,
                                        epochs=10,
                                        batch_size=500):
        self.epochs = epochs
        self.batch_size = batch_size

        if self.x_data.ndim == 2:
            scaler = StandardScaler().fit(self.x_data[train])
            self.x_data[train] = scaler.transform(self.x_data[train])

            self.x_data[test] = scaler.transform(self.x_data[test])

        self.x_data = self.x_data.reshape(self.x_data.shape[0], self.x_data.shape[1], 1)

        # defining model
        input_tensor = layers.Input(shape=(76, 1))
        convolutional_1_1 = layers.Conv1D(32, kernel_size=3, activation="relu")(input_tensor)
        max_pool_1_1 = layers.MaxPooling1D(pool_size=3)(convolutional_1_1)

        convolutional_1_2 = layers.Conv1D(32, kernel_size=5, activation="relu")(input_tensor)
        max_pool_1_2 = layers.MaxPooling1D(pool_size=3)(convolutional_1_2)

        merge_1 = layers.Concatenate(axis=1)([max_pool_1_1, max_pool_1_2])

        flatten = layers.Flatten()(merge_1)
        dense_1 = layers.Dense(64, activation='relu')(flatten)
        dropout_1 = layers.Dropout(0.5)(dense_1)

        output_tensor = layers.Dense(1, activation='sigmoid')(dropout_1)

        model = models.Model(input_tensor, output_tensor)

        # compile model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # train model
        history = model.fit(x=self.x_data[train],
                            y=self.y_data[train],
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(self.x_data[test], self.y_data[test]),
                            callbacks=[TensorBoard(log_dir='/tmp/classifier')])

        self.loss_val_index.append((np.array(history.history["val_loss"]).argmin(),
                                    np.array(history.history["val_acc"]).argmax(),
                                    np.array(history.history["acc"]).argmax()))
        self.val_accuracy_values.append(history.history['val_acc'])
        self.accuracy_values.append(history.history['acc'])

        model.summary()

        # evaluate the model
        scores = model.evaluate(self.x_data[test], self.y_data[test], verbose=0)

        print("\n--------------------------------------------------")
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        print("--------------------------------------------------\n")
        cv_scores.append(scores[1] * 100)

        if len(cv_scores) == 10:
            print("repDNA: DAC BINARY CLASSIFICATION APPROACH", file=self.filehandler)
            print("Data shape: {}".format(self.x_data.shape), file=self.filehandler)
            print("Epochs: {}, Batch size: {}".format(epochs, batch_size), file=self.filehandler)
            model.summary(print_fn=lambda x: self.filehandler.write(x + '\n'))

            # print confusion matrix
            y_pred = model.predict(self.x_data[test])
            print("Confusion matrix:",
                  confusion_matrix(y_true=self.y_data[test],
                                   y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int)),
                  file=self.filehandler)
            print("Confusion matrix:",
                  confusion_matrix(y_true=self.y_data[test],
                                   y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int)))

            # Calculate other validation scores
            conf_matrix = confusion_matrix(y_true=self.y_data[test],
                                           y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int))

            tp = conf_matrix[0, 0]
            tn = conf_matrix[1, 1]
            fp = conf_matrix[0, 1]
            fn = conf_matrix[1, 0]

            precision = tp / (tp + fp) * 100
            recall = tp/(tp + fn) * 100

            print("Recall:", recall, file=self.filehandler)
            print("Precision:", precision, file=self.filehandler)

            print("------------------------------------------------\n")

            # serialize model to JSON
            model_json = model.to_json()
            with open("../models/dac_" + self.load_file_name + "_model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("../models/dac_" + self.load_file_name + "_model.h5")
            print("Saved DAC convolutional model to disk.")

    def simple_classifier_on_repDNA_DCC(self,
                                        cv_scores,
                                        train,
                                        test,
                                        epochs=10,
                                        batch_size=500):
        self.epochs = epochs
        self.batch_size = batch_size

        if self.x_data.ndim == 2:
            scaler = StandardScaler().fit(self.x_data[train])
            self.x_data[train] = scaler.transform(self.x_data[train])

            self.x_data[test] = scaler.transform(self.x_data[test])

        self.x_data = self.x_data.reshape(self.x_data.shape[0], self.x_data.shape[1], 1)

        # defining model
        input_tensor = layers.Input(shape=(1406, 1))
        convolutional_1_1 = layers.Conv1D(32, kernel_size=3, activation="relu")(input_tensor)
        max_pool_1_1 = layers.MaxPooling1D(pool_size=3)(convolutional_1_1)

        convolutional_1_2 = layers.Conv1D(32, kernel_size=5, activation="relu")(input_tensor)
        max_pool_1_2 = layers.MaxPooling1D(pool_size=3)(convolutional_1_2)

        merge_1 = layers.Concatenate(axis=1)([max_pool_1_1, max_pool_1_2])

        flatten = layers.Flatten()(merge_1)
        dense_1 = layers.Dense(64, activation='relu')(flatten)
        dropout_1 = layers.Dropout(0.5)(dense_1)

        output_tensor = layers.Dense(1, activation='sigmoid')(dropout_1)

        model = models.Model(input_tensor, output_tensor)

        # compile model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # train model
        history = model.fit(x=self.x_data[train],
                            y=self.y_data[train],
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(self.x_data[test], self.y_data[test]),
                            callbacks=[TensorBoard(log_dir='/tmp/classifier')])

        self.loss_val_index.append((np.array(history.history["val_loss"]).argmin(),
                                    np.array(history.history["val_acc"]).argmax(),
                                    np.array(history.history["acc"]).argmax()))
        self.val_accuracy_values.append(history.history['val_acc'])
        self.accuracy_values.append(history.history['acc'])

        model.summary()

        # evaluate the model
        scores = model.evaluate(self.x_data[test], self.y_data[test], verbose=0)

        print("\n--------------------------------------------------")
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        print("--------------------------------------------------\n")
        cv_scores.append(scores[1] * 100)

        if len(cv_scores) == 10:
            print("repDNA: DCC BINARY CLASSIFICATION APPROACH", file=self.filehandler)
            print("Data shape: {}".format(self.x_data.shape), file=self.filehandler)
            print("Epochs: {}, Batch size: {}".format(epochs, batch_size), file=self.filehandler)
            model.summary(print_fn=lambda x: self.filehandler.write(x + '\n'))

            # print confusion matrix
            y_pred = model.predict(self.x_data[test])
            print("Confusion matrix:",
                  confusion_matrix(y_true=self.y_data[test],
                                   y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int)),
                  file=self.filehandler)
            print("Confusion matrix:",
                  confusion_matrix(y_true=self.y_data[test],
                                   y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int)))

            # Calculate other validation scores
            conf_matrix = confusion_matrix(y_true=self.y_data[test],
                                           y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int))

            tp = conf_matrix[0, 0]
            tn = conf_matrix[1, 1]
            fp = conf_matrix[0, 1]
            fn = conf_matrix[1, 0]

            precision = tp / (tp + fp) * 100
            recall = tp/(tp + fn) * 100

            print("Recall:", recall, file=self.filehandler)
            print("Precision:", precision, file=self.filehandler)

            print("------------------------------------------------\n")

            # serialize model to JSON
            model_json = model.to_json()
            with open("../models/dcc_" + self.load_file_name + "_model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("../models/dcc_" + self.load_file_name + "_model.h5")
            print("Saved DCC convolutional model to disk.")

    def simple_classifier_on_repDNA_PC_PseDNC(self,
                                              cv_scores,
                                              train,
                                              test,
                                              epochs=10,
                                              batch_size=500):

        self.epochs = epochs
        self.batch_size = batch_size

        if self.x_data.ndim == 2:
            scaler = StandardScaler().fit(self.x_data[train])
            self.x_data[train] = scaler.transform(self.x_data[train])

            self.x_data[test] = scaler.transform(self.x_data[test])

        self.x_data = self.x_data.reshape(self.x_data.shape[0], self.x_data.shape[1], 1)

        # defining model
        input_tensor = layers.Input(shape=(18, 1))
        convolutional_1_1 = layers.Conv1D(32, kernel_size=3, activation="relu")(input_tensor)
        convolutional_1_2 = layers.Conv1D(32, kernel_size=4, activation="relu")(input_tensor)
        convolutional_1_3 = layers.Conv1D(32, kernel_size=5, activation="relu")(input_tensor)

        merge_1 = layers.Concatenate(axis=1)([convolutional_1_1, convolutional_1_2, convolutional_1_3])

        flatten = layers.Flatten()(merge_1)
        dense_1 = layers.Dense(64, activation='relu')(flatten)
        dropout_1 = layers.Dropout(0.5)(dense_1)

        output_tensor = layers.Dense(1, activation='sigmoid')(dropout_1)

        model = models.Model(input_tensor, output_tensor)

        # compile model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # train model
        history = model.fit(x=self.x_data[train],
                            y=self.y_data[train],
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(self.x_data[test], self.y_data[test]),
                            callbacks=[TensorBoard(log_dir='/tmp/classifier')])

        self.loss_val_index.append((np.array(history.history["val_loss"]).argmin(),
                                    np.array(history.history["val_acc"]).argmax(),
                                    np.array(history.history["acc"]).argmax()))
        self.val_accuracy_values.append(history.history['val_acc'])
        self.accuracy_values.append(history.history['acc'])

        model.summary()

        # evaluate the model
        scores = model.evaluate(self.x_data[test], self.y_data[test], verbose=0)

        print("\n--------------------------------------------------")
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        print("--------------------------------------------------\n")
        cv_scores.append(scores[1] * 100)

        if len(cv_scores) == 10:
            print("repDNA: PC-PseDNC BINARY CLASSIFICATION APPROACH", file=self.filehandler)
            print("Data shape: {}".format(self.x_data.shape), file=self.filehandler)
            print("Epochs: {}, Batch size: {}".format(epochs, batch_size), file=self.filehandler)
            model.summary(print_fn=lambda x: self.filehandler.write(x + '\n'))

            # print confusion matrix
            y_pred = model.predict(self.x_data[test])
            print("Confusion matrix:",
                  confusion_matrix(y_true=self.y_data[test],
                                   y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int)),
                  file=self.filehandler)
            print("Confusion matrix:",
                  confusion_matrix(y_true=self.y_data[test],
                                   y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int)))

            # Calculate other validation scores
            conf_matrix = confusion_matrix(y_true=self.y_data[test],
                                           y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int))

            tp = conf_matrix[0, 0]
            tn = conf_matrix[1, 1]
            fp = conf_matrix[0, 1]
            fn = conf_matrix[1, 0]

            precision = tp / (tp + fp) * 100
            recall = tp/(tp + fn) * 100

            print("Recall:", recall, file=self.filehandler)
            print("Precision:", precision, file=self.filehandler)

            print("------------------------------------------------\n")

            # serialize model to JSON
            model_json = model.to_json()
            with open("../models/PC_PseDNC_" + self.load_file_name + "_model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("../models/PC_PseDNC_" + self.load_file_name + "_model.h5")
            print("Saved PC-PseDNC convolutional model to disk.")

    def simple_classifier_on_repDNA_PC_PseTNC(self,
                                              cv_scores,
                                              train,
                                              test,
                                              epochs=10,
                                              batch_size=500):

        self.epochs = epochs
        self.batch_size = batch_size

        if self.x_data.ndim == 2:
            scaler = StandardScaler().fit(self.x_data[train])
            self.x_data[train] = scaler.transform(self.x_data[train])

            self.x_data[test] = scaler.transform(self.x_data[test])

        self.x_data = self.x_data.reshape(self.x_data.shape[0], self.x_data.shape[1], 1)

        # defining model
        input_tensor = layers.Input(shape=(66, 1))
        convolutional_1_1 = layers.Conv1D(32, kernel_size=3, activation="relu")(input_tensor)
        convolutional_1_2 = layers.Conv1D(32, kernel_size=4, activation="relu")(input_tensor)
        convolutional_1_3 = layers.Conv1D(32, kernel_size=5, activation="relu")(input_tensor)

        merge_1 = layers.Concatenate(axis=1)([convolutional_1_1, convolutional_1_2, convolutional_1_3])

        flatten = layers.Flatten()(merge_1)
        dense_1 = layers.Dense(64, activation='relu')(flatten)
        dropout_1 = layers.Dropout(0.5)(dense_1)

        output_tensor = layers.Dense(1, activation='sigmoid')(dropout_1)

        model = models.Model(input_tensor, output_tensor)

        # compile model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # train model
        history = model.fit(x=self.x_data[train],
                            y=self.y_data[train],
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(self.x_data[test], self.y_data[test]),
                            callbacks=[TensorBoard(log_dir='/tmp/classifier')])

        self.loss_val_index.append((np.array(history.history["val_loss"]).argmin(),
                                    np.array(history.history["val_acc"]).argmax(),
                                    np.array(history.history["acc"]).argmax()))
        self.val_accuracy_values.append(history.history['val_acc'])
        self.accuracy_values.append(history.history['acc'])

        model.summary()

        # evaluate the model
        scores = model.evaluate(self.x_data[test], self.y_data[test], verbose=0)

        print("\n--------------------------------------------------")
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        print("--------------------------------------------------\n")
        cv_scores.append(scores[1] * 100)

        if len(cv_scores) == 10:
            print("repDNA: PC-PseTNC BINARY CLASSIFICATION APPROACH", file=self.filehandler)
            print("Data shape: {}".format(self.x_data.shape), file=self.filehandler)
            print("Epochs: {}, Batch size: {}".format(epochs, batch_size), file=self.filehandler)
            model.summary(print_fn=lambda x: self.filehandler.write(x + '\n'))

            # print confusion matrix
            y_pred = model.predict(self.x_data[test])
            print("Confusion matrix:",
                  confusion_matrix(y_true=self.y_data[test],
                                   y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int)),
                  file=self.filehandler)
            print("Confusion matrix:",
                  confusion_matrix(y_true=self.y_data[test],
                                   y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int)))

            # Calculate other validation scores
            conf_matrix = confusion_matrix(y_true=self.y_data[test],
                                           y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int))

            tp = conf_matrix[0, 0]
            tn = conf_matrix[1, 1]
            fp = conf_matrix[0, 1]
            fn = conf_matrix[1, 0]

            precision = tp / (tp + fp) * 100
            recall = tp/(tp + fn) * 100

            print("Recall:", recall, file=self.filehandler)
            print("Precision:", precision, file=self.filehandler)

            print("------------------------------------------------\n")

            # serialize model to JSON
            model_json = model.to_json()
            with open("../models/PC_PseTNC_" + self.load_file_name + "_model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("../models/PC_PseTNC_" + self.load_file_name + "_model.h5")
            print("Saved PC-PseTNC convolutional model to disk.")

    def simple_classifier_on_repDNA_SC_PseDNC(self,
                                              cv_scores,
                                              train,
                                              test,
                                              epochs=10,
                                              batch_size=500):

        self.epochs = epochs
        self.batch_size = batch_size

        if self.x_data.ndim == 2:
            scaler = StandardScaler().fit(self.x_data[train])
            self.x_data[train] = scaler.transform(self.x_data[train])

            self.x_data[test] = scaler.transform(self.x_data[test])

        self.x_data = self.x_data.reshape(self.x_data.shape[0], self.x_data.shape[1], 1)

        # defining model
        input_tensor = layers.Input(shape=(92, 1))
        convolutional_1_1 = layers.Conv1D(32, kernel_size=3, activation="relu")(input_tensor)
        convolutional_1_2 = layers.Conv1D(32, kernel_size=4, activation="relu")(input_tensor)
        convolutional_1_3 = layers.Conv1D(32, kernel_size=5, activation="relu")(input_tensor)

        merge_1 = layers.Concatenate(axis=1)([convolutional_1_1, convolutional_1_2, convolutional_1_3])

        flatten = layers.Flatten()(merge_1)
        dense_1 = layers.Dense(64, activation='relu')(flatten)
        dropout_1 = layers.Dropout(0.5)(dense_1)

        output_tensor = layers.Dense(1, activation='sigmoid')(dropout_1)

        model = models.Model(input_tensor, output_tensor)

        # compile model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # train model
        history = model.fit(x=self.x_data[train],
                            y=self.y_data[train],
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(self.x_data[test], self.y_data[test]),
                            callbacks=[TensorBoard(log_dir='/tmp/classifier')])

        self.loss_val_index.append((np.array(history.history["val_loss"]).argmin(),
                                    np.array(history.history["val_acc"]).argmax(),
                                    np.array(history.history["acc"]).argmax()))
        self.val_accuracy_values.append(history.history['val_acc'])
        self.accuracy_values.append(history.history['acc'])

        model.summary()

        # evaluate the model
        scores = model.evaluate(self.x_data[test], self.y_data[test], verbose=0)

        print("\n--------------------------------------------------")
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        print("--------------------------------------------------\n")
        cv_scores.append(scores[1] * 100)

        if len(cv_scores) == 10:
            print("repDNA: SC-PseDNC BINARY CLASSIFICATION APPROACH", file=self.filehandler)
            print("Data shape: {}".format(self.x_data.shape), file=self.filehandler)
            print("Epochs: {}, Batch size: {}".format(epochs, batch_size), file=self.filehandler)
            model.summary(print_fn=lambda x: self.filehandler.write(x + '\n'))

            # print confusion matrix
            y_pred = model.predict(self.x_data[test])
            print("Confusion matrix:",
                  confusion_matrix(y_true=self.y_data[test],
                                   y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int)),
                  file=self.filehandler)
            print("Confusion matrix:",
                  confusion_matrix(y_true=self.y_data[test],
                                   y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int)))

            # Calculate other validation scores
            conf_matrix = confusion_matrix(y_true=self.y_data[test],
                                           y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int))

            tp = conf_matrix[0, 0]
            tn = conf_matrix[1, 1]
            fp = conf_matrix[0, 1]
            fn = conf_matrix[1, 0]

            precision = tp / (tp + fp) * 100
            recall = tp/(tp + fn) * 100

            print("Recall:", recall, file=self.filehandler)
            print("Precision:", precision, file=self.filehandler)

            print("------------------------------------------------\n")

            # serialize model to JSON
            model_json = model.to_json()
            with open("../models/SC_PseDNC_" + self.load_file_name + "_model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("../models/SC_PseDNC_" + self.load_file_name + "_model.h5")
            print("Saved SC-PseDNC convolutional model to disk.")

    def simple_classifier_on_repDNA_SC_PseTNC(self,
                                              cv_scores,
                                              train,
                                              test,
                                              epochs=10,
                                              batch_size=500):

        self.epochs = epochs
        self.batch_size = batch_size

        if self.x_data.ndim == 2:
            scaler = StandardScaler().fit(self.x_data[train])
            self.x_data[train] = scaler.transform(self.x_data[train])

            self.x_data[test] = scaler.transform(self.x_data[test])

        self.x_data = self.x_data.reshape(self.x_data.shape[0], self.x_data.shape[1], 1)

        # defining model
        input_tensor = layers.Input(shape=(88, 1))
        convolutional_1_1 = layers.Conv1D(32, kernel_size=3, activation="relu")(input_tensor)
        convolutional_1_2 = layers.Conv1D(32, kernel_size=4, activation="relu")(input_tensor)
        convolutional_1_3 = layers.Conv1D(32, kernel_size=5, activation="relu")(input_tensor)

        merge_1 = layers.Concatenate(axis=1)([convolutional_1_1, convolutional_1_2, convolutional_1_3])

        flatten = layers.Flatten()(merge_1)
        dense_1 = layers.Dense(64, activation='relu')(flatten)
        dropout_1 = layers.Dropout(0.5)(dense_1)

        output_tensor = layers.Dense(1, activation='sigmoid')(dropout_1)

        model = models.Model(input_tensor, output_tensor)

        # compile model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # train model
        history = model.fit(x=self.x_data[train],
                            y=self.y_data[train],
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(self.x_data[test], self.y_data[test]),
                            callbacks=[TensorBoard(log_dir='/tmp/classifier')])

        self.loss_val_index.append((np.array(history.history["val_loss"]).argmin(),
                                    np.array(history.history["val_acc"]).argmax(),
                                    np.array(history.history["acc"]).argmax()))
        self.val_accuracy_values.append(history.history['val_acc'])
        self.accuracy_values.append(history.history['acc'])

        model.summary()

        # evaluate the model
        scores = model.evaluate(self.x_data[test], self.y_data[test], verbose=0)

        print("\n--------------------------------------------------")
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        print("--------------------------------------------------\n")
        cv_scores.append(scores[1] * 100)

        if len(cv_scores) == 10:
            print("repDNA: SC-PseTNC BINARY CLASSIFICATION APPROACH", file=self.filehandler)
            print("Data shape: {}".format(self.x_data.shape), file=self.filehandler)
            print("Epochs: {}, Batch size: {}".format(epochs, batch_size), file=self.filehandler)
            model.summary(print_fn=lambda x: self.filehandler.write(x + '\n'))

            # print confusion matrix
            y_pred = model.predict(self.x_data[test])
            print("Confusion matrix:",
                  confusion_matrix(y_true=self.y_data[test], y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int)),
                  file=self.filehandler)
            print("Confusion matrix:",
                  confusion_matrix(y_true=self.y_data[test], y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int)))

            # Calculate other validation scores
            conf_matrix = confusion_matrix(y_true=self.y_data[test],
                                           y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int))

            tp = conf_matrix[0, 0]
            tn = conf_matrix[1, 1]
            fp = conf_matrix[0, 1]
            fn = conf_matrix[1, 0]

            precision = tp / (tp + fp) * 100
            recall = tp/(tp + fn) * 100

            print("Recall:", recall, file=self.filehandler)
            print("Precision:", precision, file=self.filehandler)

            print("------------------------------------------------\n")

            # serialize model to JSON
            model_json = model.to_json()
            with open("../models/SC_PseTNC_" + self.load_file_name + "_model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("../models/SC_PseTNC_" + self.load_file_name + "_model.h5")
            print("Saved SC-PseTNC convolutional model to disk.")

    def overall_classifier(self,
                           cv_scores,
                           train,
                           test,
                           epochs=10,
                           batch_size=100):
        print("Starting overall classifier")
        # set parameter for training
        self.epochs = epochs
        self.batch_size = batch_size

        # Read simple data
        x_data_simple = self.x_data.reshape(self.x_data.shape + (1,))

        # Read DiProDB data
        print("Reading DiProDB data...")
        dataset = "dint"
        x_data_DiProDB = np.load(file="../data/x_" + dataset + ("_" if len(dataset)!=0 else "")+ self.load_file_name + "_" + str(self.samples_per_file) + "_samples" +
                              ("_" + str(self.pre_length) + "_pre" if self.pre_length!=0 else "") + ("_" + str(self.post_length) + "_post" if self.post_length!=0 else "") + ".npy")

        x_data_DiProDB = x_data_DiProDB.reshape(x_data_DiProDB.shape + (1,))

        # Read repDNA data
        self.pre_length = 0
        self.post_length = 0

        print("Reading Kmer data...")
        dataset = "kmer"
        x_data_kmer = np.load(file="../data/x_" + dataset + ("_" if len(dataset)!=0 else "")+ self.load_file_name + "_" + str(self.samples_per_file) + "_samples" +
                                   ("_" + str(self.pre_length) + "_pre" if self.pre_length!=0 else "") + ("_" + str(self.post_length) + "_post" if self.post_length!=0 else "") + ".npy")
        x_data_kmer = x_data_kmer.reshape(x_data_kmer.shape + (1,))

        print("Reading IDkmer data...")
        dataset = "IDkmer"
        x_data_IDkmer = np.load(file="../data/x_" + dataset + ("_" if len(dataset)!=0 else "")+ self.load_file_name + "_" + str(self.samples_per_file) + "_samples" +
                                   ("_" + str(self.pre_length) + "_pre" if self.pre_length!=0 else "") + ("_" + str(self.post_length) + "_post" if self.post_length!=0 else "") + ".npy")
        x_data_IDkmer = x_data_IDkmer.reshape(x_data_IDkmer.shape + (1,))

        print("Reading DAC data...")
        dataset = "dac"
        x_data_dac = np.load(file="../data/x_" + dataset + ("_" if len(dataset)!=0 else "")+ self.load_file_name + "_" + str(self.samples_per_file) + "_samples" +
                                   ("_" + str(self.pre_length) + "_pre" if self.pre_length!=0 else "") + ("_" + str(self.post_length) + "_post" if self.post_length!=0 else "") + ".npy")
        x_data_dac = x_data_dac.reshape(x_data_dac.shape + (1,))

        print("Reading DCC data...")
        dataset = "dcc"
        x_data_dcc = np.load(file="../data/x_" + dataset + ("_" if len(dataset)!=0 else "")+ self.load_file_name + "_" + str(self.samples_per_file) + "_samples" +
                                   ("_" + str(self.pre_length) + "_pre" if self.pre_length!=0 else "") + ("_" + str(self.post_length) + "_post" if self.post_length!=0 else "") + ".npy")
        x_data_dcc = x_data_dcc.reshape(x_data_dcc.shape + (1,))

        print("Reading PC-PseDNC data...")
        dataset = "PC_PseDNC"
        x_data_PC_PseDNC = np.load(file="../data/x_" + dataset + ("_" if len(dataset)!=0 else "")+ self.load_file_name + "_" + str(self.samples_per_file) + "_samples" +
                                   ("_" + str(self.pre_length) + "_pre" if self.pre_length!=0 else "") + ("_" + str(self.post_length) + "_post" if self.post_length!=0 else "") + ".npy")
        x_data_PC_PseDNC = x_data_PC_PseDNC.reshape(x_data_PC_PseDNC.shape + (1,))

        print("Reading PC-PseTNC data...")
        dataset = "PC_PseTNC"
        x_data_PC_PseTNC = np.load(file="../data/x_" + dataset + ("_" if len(dataset)!=0 else "")+ self.load_file_name + "_" + str(self.samples_per_file) + "_samples" +
                                   ("_" + str(self.pre_length) + "_pre" if self.pre_length!=0 else "") + ("_" + str(self.post_length) + "_post" if self.post_length!=0 else "") + ".npy")
        x_data_PC_PseTNC = x_data_PC_PseTNC.reshape(x_data_PC_PseTNC.shape + (1,))

        print("Reading SC-PseDNC data...")
        dataset = "SC_PseDNC"
        x_data_SC_PseDNC = np.load(file="../data/x_" + dataset + ("_" if len(dataset)!=0 else "")+ self.load_file_name + "_" + str(self.samples_per_file) + "_samples" +
                                   ("_" + str(self.pre_length) + "_pre" if self.pre_length!=0 else "") + ("_" + str(self.post_length) + "_post" if self.post_length!=0 else "") + ".npy")
        x_data_SC_PseDNC = x_data_SC_PseDNC.reshape(x_data_SC_PseDNC.shape + (1,))

        print("Reading SC-PseTNC data...")
        dataset = "SC_PseTNC"
        x_data_SC_PseTNC = np.load(file="../data/x_" + dataset + ("_" if len(dataset)!=0 else "")+ self.load_file_name + "_" + str(self.samples_per_file) + "_samples" +
                                   ("_" + str(self.pre_length) + "_pre" if self.pre_length!=0 else "") + ("_" + str(self.post_length) + "_post" if self.post_length!=0 else "") + ".npy")
        x_data_SC_PseTNC = x_data_SC_PseTNC.reshape(x_data_SC_PseTNC.shape + (1,))

        self.pre_length = 300
        self.post_length = 300
        print("Finished reading data.")

        # Truncate and prepare models
        print("Preparing models...")
        print("Loading Simple model...")
        classifier_json_file = None
        with open("../models/simple_" + self.load_file_name + "_model.json") as fh:
            classifier_json_file = fh.read()
        simple_classifier_model = model_from_json(classifier_json_file)
        simple_classifier_model.load_weights("../models/simple_" + self.load_file_name + "_model.h5")

        for layer in simple_classifier_model.layers:
            layer.name = "simple_" + layer.name
            layer.trainable = False
        for i in range(5):
            simple_classifier_model.layers.pop()

        print("Loading DiProDB model...")
        with open("../models/DiProDB_" + self.load_file_name + "_model.json") as fh:
            classifier_json_file = fh.read()
        DiProDB_classifier_model = model_from_json(classifier_json_file)
        DiProDB_classifier_model.load_weights("../models/DiProDB_" + self.load_file_name + "_model.h5")

        for layer in DiProDB_classifier_model.layers:
            layer.name = "DiProDB_" + layer.name
            layer.trainable = False
        for i in range(5):
            DiProDB_classifier_model.layers.pop()

        print("Loading IDkmer model...")
        with open("../models/IDkmer_" + self.load_file_name + "_model.json") as fh:
            classifier_json_file = fh.read()
        IDkmer_classifier_model = model_from_json(classifier_json_file)
        IDkmer_classifier_model.load_weights("../models/IDkmer_" + self.load_file_name + "_model.h5")

        for layer in IDkmer_classifier_model.layers:
            layer.name = "IDkmer_" + layer.name
            layer.trainable = False
        for i in range(3):
            IDkmer_classifier_model.layers.pop()

        print("Loading DAC model...")
        with open("../models/dac_" + self.load_file_name + "_model.json") as fh:
            classifier_json_file = fh.read()
        dac_classifier_model = model_from_json(classifier_json_file)
        dac_classifier_model.load_weights("../models/dac_" + self.load_file_name + "_model.h5")

        for layer in dac_classifier_model.layers:
            layer.name = "DAC_" + layer.name
            layer.trainable = False
        for i in range(3):
            dac_classifier_model.layers.pop()

        print("Loading DCC model...")
        with open("../models/dcc_" + self.load_file_name + "_model.json") as fh:
            classifier_json_file = fh.read()
        dcc_classifier_model = model_from_json(classifier_json_file)
        dcc_classifier_model.load_weights("../models/dcc_" + self.load_file_name + "_model.h5")

        for layer in dcc_classifier_model.layers:
            layer.name = "DCC_" + layer.name
            layer.trainable = False
        for i in range(3):
            dcc_classifier_model.layers.pop()

        print("Loading PC-PseDNC model...")
        with open("../models/PC_PseDNC_" + self.load_file_name + "_model.json") as fh:
            classifier_json_file = fh.read()
        PC_PseDNC_classifier_model = model_from_json(classifier_json_file)
        PC_PseDNC_classifier_model.load_weights("../models/PC_PseDNC_" + self.load_file_name + "_model.h5")

        for layer in PC_PseDNC_classifier_model.layers:
            layer.name = "PC_PseDNC_" + layer.name
            layer.trainable = False
        for i in range(3):
            PC_PseDNC_classifier_model.layers.pop()

        print("Loading PC-PseTNC model...")
        with open("../models/PC_PseTNC_" + self.load_file_name + "_model.json") as fh:
            classifier_json_file = fh.read()
        PC_PseTNC_classifier_model = model_from_json(classifier_json_file)
        PC_PseTNC_classifier_model.load_weights("../models/PC_PseTNC_" + self.load_file_name + "_model.h5")

        for layer in PC_PseTNC_classifier_model.layers:
            layer.name = "PC_PseTNC_" + layer.name
            layer.trainable = False
        for i in range(3):
            PC_PseTNC_classifier_model.layers.pop()

        print("Loading SC-PseDNC model...")
        with open("../models/SC_PseDNC_" + self.load_file_name + "_model.json") as fh:
            classifier_json_file = fh.read()
        SC_PseDNC_classifier_model = model_from_json(classifier_json_file)
        SC_PseDNC_classifier_model.load_weights("../models/SC_PseDNC_" + self.load_file_name + "_model.h5")

        for layer in SC_PseDNC_classifier_model.layers:
            layer.name = "SC_PseDNC_" + layer.name
            layer.trainable = False
        for i in range(3):
            SC_PseDNC_classifier_model.layers.pop()

        print("Loading SC-PseTNC model...")
        with open("../models/SC_PseTNC_" + self.load_file_name + "_model.json") as fh:
            classifier_json_file = fh.read()
        SC_PseTNC_classifier_model = model_from_json(classifier_json_file)
        SC_PseTNC_classifier_model.load_weights("../models/SC_PseTNC_" + self.load_file_name + "_model.h5")

        for layer in SC_PseTNC_classifier_model.layers:
            layer.name = "SC_PseTNC_" + layer.name
            layer.trainable = False
        for i in range(3):
            SC_PseTNC_classifier_model.layers.pop()

        print("Finished truncating models.")

        print("Building model...")
        simple_input_tensor = simple_classifier_model.layers[0]
        DiProDB_input_tensor = DiProDB_classifier_model.layers[0]
        IDkmer_input_tensor = IDkmer_classifier_model.layers[0]
        DAC_input_tensor = dac_classifier_model.layers[0]
        DCC_input_tensor = dcc_classifier_model.layers[0]
        PC_PseDNC_input_tensor = PC_PseDNC_classifier_model.layers[0]
        PC_PseTNC_input_tensor = PC_PseTNC_classifier_model.layers[0]
        SC_PseDNC_input_tensor = SC_PseDNC_classifier_model.layers[0]
        SC_PseTNC_input_tensor = SC_PseTNC_classifier_model.layers[0]


        concat = layers.concatenate([simple_classifier_model.layers[-1].output,
                                     DiProDB_classifier_model.layers[-1].output,
                                     # IDkmer_classifier_model.layers[-1].output,
                                     # dac_classifier_model.layers[-1].output,
                                     # dcc_classifier_model.layers[-1].output,
                                     # PC_PseDNC_classifier_model.layers[-1].output,
                                     # PC_PseTNC_classifier_model.layers[-1].output,
                                     # SC_PseDNC_classifier_model.layers[-1].output,
                                     # SC_PseTNC_classifier_model.layers[-1].output
                                     ])

        dense_1 = layers.Dense(1024, activation='relu')(concat)
        output_tensor = layers.Dense(1, activation='sigmoid')(dense_1)

        model = models.Model(inputs=[simple_input_tensor.input,
                                     DiProDB_input_tensor.input,
                                     # IDkmer_input_tensor.input,
                                     # DAC_input_tensor.input,
                                     # DCC_input_tensor.input,
                                     # PC_PseDNC_input_tensor.input,
                                     # PC_PseTNC_input_tensor.input,
                                     # SC_PseDNC_input_tensor.input,
                                     # SC_PseTNC_input_tensor.input
                                     ],
                             outputs=[output_tensor])

        # compile model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # train model
        history = model.fit(x=[x_data_simple[train],
                               x_data_DiProDB[train],
                               # x_data_IDkmer[train],
                               # x_data_dac[train],
                               # x_data_dcc[train],
                               # x_data_PC_PseDNC[train],
                               # x_data_PC_PseTNC[train],
                               # x_data_SC_PseDNC[train],
                               # x_data_SC_PseTNC[train]
                               ],
                            y=[self.y_data[train]],
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=([x_data_simple[test],
                                              x_data_DiProDB[test],
                                              # x_data_IDkmer[test],
                                              # x_data_dac[test],
                                              # x_data_dcc[test],
                                              # x_data_PC_PseDNC[test],
                                              # x_data_PC_PseTNC[test],
                                              # x_data_SC_PseDNC[test],
                                              # x_data_SC_PseTNC[test]
                                              ],
                                             [self.y_data[test]]),
                            callbacks=[TensorBoard(log_dir='/tmp/classifier')])

        self.loss_val_index.append((np.array(history.history["val_loss"]).argmin(),
                                    np.array(history.history["val_acc"]).argmax(),
                                    np.array(history.history["acc"]).argmax()))
        self.val_accuracy_values.append(history.history['val_acc'])
        self.accuracy_values.append(history.history['acc'])

        model.summary()

        # evaluate the model
        scores = model.evaluate([x_data_simple[test],
                                 x_data_DiProDB[test],
                                 # x_data_IDkmer[test],
                                 # x_data_dac[test],
                                 # x_data_dcc[test],
                                 # x_data_PC_PseDNC[test],
                                 # x_data_PC_PseTNC[test],
                                 # x_data_SC_PseDNC[test],
                                 # x_data_SC_PseTNC[test]
                                 ],
                                [self.y_data[test]],
                                verbose=0)


        print("\n--------------------------------------------------")
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        print("--------------------------------------------------\n")
        cv_scores.append(scores[1] * 100)

        if len(cv_scores) == 10:
            print("OVERALL BINARY CLASSIFICATION APPROACH", file=self.filehandler)
            print("Data shape: {}".format(self.x_data.shape), file=self.filehandler)
            print("Epochs: {}, Batch size: {}".format(epochs, batch_size), file=self.filehandler)
            model.summary(print_fn=lambda x: self.filehandler.write(x + '\n'))

            # print confusion matrix
            y_pred = model.predict([x_data_simple[test],
                                    x_data_DiProDB[test],
                                    # x_data_IDkmer[test],
                                    # x_data_dac[test],
                                    # x_data_dcc[test],
                                    # x_data_PC_PseDNC[test],
                                    # x_data_PC_PseTNC[test],
                                    # x_data_SC_PseDNC[test],
                                    # x_data_SC_PseTNC[test]
                                    ])


            conf_matrix = confusion_matrix(y_true=self.y_data[test],
                                           y_pred=(y_pred > 0.5).astype(int)[:, 0])

            print("Confusion matrix:", conf_matrix, file=self.filehandler)
            print("Confusion matrix:", conf_matrix)

            # Calculate other validation scores

            tp = conf_matrix[0, 0]
            tn = conf_matrix[1, 1]
            fp = conf_matrix[0, 1]
            fn = conf_matrix[1, 0]

            precision = tp / (tp + fp) * 100
            recall = tp/(tp + fn) * 100

            print("Recall:", recall, file=self.filehandler)
            print("Precision:", precision, file=self.filehandler)

            print("------------------------------------------------\n")

            # serialize model to JSON
            model_json = model.to_json()
            with open("../models/overall_" + self.load_file_name + "_model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("../models/overall_" + self.load_file_name + "_model.h5")
            print("Saved overall convolutional model to disk.")
