import numpy as np

from keras import models
from keras import layers
from keras.callbacks import TensorBoard
from keras import backend
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier


class Model:

    def __init__(self,
                 x_data,
                 y_data,
                 filehandler,
                 pre_length=300,
                 post_length=300):
        self.x_data = x_data
        self.y_data = y_data
        self.filehandler = filehandler
        self.pre_length = pre_length
        self.post_length = post_length

        # Find best epoch
        self.loss_val_index = []
        self.accuracy_values = []
        self.val_accuracy_values = []

        self.epochs = None
        self.batch_size = None

        config = tf.ConfigProto( device_count = {'GPU': 0 , 'CPU': 4} )
        sess = tf.Session(config=config)
        backend.set_session(sess)


    def normalize_labels(self):
        return self.x_data.argmax(axis=2)*2/3 - 1

    def simple_classifier(self,
                          cv_scores,
                          train,
                          test,
                          epochs=10,
                          batch_size=500):

        self.epochs = epochs
        self.batch_size = batch_size

        self.x_data = self.x_data.reshape((self.x_data.shape[0], self.x_data.shape[1], self.x_data.shape[2], 1))

        # defining model
        input_tensor = layers.Input(shape=(self.pre_length + 2 + self.post_length, 4, 1))
        convolutional_1 = layers.Conv2D(32, kernel_size=(3, 4), input_shape=(602, 4, 1))(input_tensor)
        max_pool_1 = layers.MaxPooling2D((2, 1))(convolutional_1)
        convolutional_2 = layers.Conv2D(64, kernel_size=(3, 1))(max_pool_1)
        max_pool_2 = layers.MaxPooling2D((3, 1))(convolutional_2)
        convolutional_3 = layers.Conv2D(128, kernel_size=(5,1))(max_pool_2)
        max_pool_3 = layers.MaxPooling2D(3,1)(convolutional_3)
        flatten = layers.Flatten()(max_pool_3)
        dense_1 = layers.Dense(64, activation='relu')(flatten)
        dropout_1 = layers.Dropout(0.5)(dense_1)
        dense_2 = layers.Dense(64, activation='relu')(dropout_1)
        dropout_2 = layers.Dropout(0.5)(dense_2)
        output_tensor = layers.Dense(1, activation='sigmoid')(dropout_2)

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
                            validation_data=(self.x_data[test],self.y_data[test]),
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
                  confusion_matrix(y_true=self.y_data[test], y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int)),
                  file=self.filehandler)
            print("Confusion matrix:",
                  confusion_matrix(y_true=self.y_data[test], y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int)))

            # Calculate other validation scores
            conf_matrix = confusion_matrix(y_true=self.y_data[test], y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int))

            tp = conf_matrix[0, 0]
            tn = conf_matrix[1, 1]
            fp = conf_matrix[0, 1]
            fn = conf_matrix[1, 0]

            precision = tp / (tp + fp) * 100
            recall = tp/(tp + fn) * 100

            print("Recall:", recall, file=self.filehandler)
            print("Precision:",precision, file=self.filehandler)

            print("------------------------------------------------\n")

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
                                     epochs=10,
                                     batch_size=500):
        self.epochs = epochs
        self.batch_size = batch_size

        # defining model
        input_tensor = layers.Input(shape=(self.pre_length + 2 + self.post_length - 1, 15, 1))
        convolutional_1 = layers.Conv2D(32, kernel_size=(5, 15), input_shape=(601, 15, 1), activation="relu")(input_tensor)
        max_pool_1 = layers.MaxPooling2D((3,1))(convolutional_1)
        convolutional_2 = layers.Conv2D(64, kernel_size=(3, 1), activation="relu")(max_pool_1)
        max_pool_2 = layers.MaxPooling2D((3,1))(convolutional_2)
        flatten = layers.Flatten()(max_pool_2)
        dense_1 = layers.Dense(64, activation='tanh')(flatten)
        dropout_1 = layers.Dropout(0.5)(dense_1)
        dense_2 = layers.Dense(64, activation='relu')(dropout_1)
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
                            validation_data=(self.x_data[test],self.y_data[test]),
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
                  confusion_matrix(y_true=self.y_data[test], y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int)),
                  file=self.filehandler)
            print("Confusion matrix:",
                  confusion_matrix(y_true=self.y_data[test], y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int)))

            # Calculate other validation scores
            conf_matrix = confusion_matrix(y_true=self.y_data[test], y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int))

            tp = conf_matrix[0, 0]
            tn = conf_matrix[1, 1]
            fp = conf_matrix[0, 1]
            fn = conf_matrix[1, 0]

            precision = tp / (tp + fp) * 100
            recall = tp/(tp + fn) * 100

            print("Recall:", recall, file=self.filehandler)
            print("Precision:",precision, file=self.filehandler)

            print("------------------------------------------------\n")
