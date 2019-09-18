import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.callbacks import TensorBoard
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import confusion_matrix


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

    def simple_classifier(self,
                          cv_scores,
                          train,
                          test,
                          epochs=10,
                          batch_size=500):

        # defining model
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(100, input_shape=(self.pre_length + 2 + self.post_length, 4), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(30, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        # compile model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # train model
        model.fit(x=self.x_data[train],
                  y=self.y_data[train],
                  epochs=epochs,
                  batch_size=batch_size,
                  callbacks=[TensorBoard(log_dir='/tmp/classifier')])

        model.summary()

        # evaluate the model
        scores = model.evaluate(self.x_data[test], self.y_data[test], verbose=0)

        print("\n--------------------------------------------------")
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        print("--------------------------------------------------\n")
        cv_scores.append(scores[1] * 100)

        if len(cv_scores) == 10:
            print("BINARY CLASSIFICATION APPROACH", file=self.filehandler)
            print("Data shape: {}".format(self.x_data.shape), file=self.filehandler)
            model.summary(print_fn=lambda x: self.filehandler.write(x + '\n'))

            # print confusion matrix
            y_pred = model.predict(self.x_data[test])
            print("Confusion matrix:",
                  confusion_matrix(y_true=self.y_data[test], y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int)),
                  file=self.filehandler)
            print("Confusion matrix:",
                  confusion_matrix(y_true=self.y_data[test], y_pred=(y_pred.reshape((len(y_pred))) > 0.5).astype(int)))
            print("------------------------------------------------\n")
            
    def multi_label_classifier(self,
                               cv_scores,
                               train,
                               test,
                               epochs=10,
                               batch_size=500):
        
        onehot_encoder = OneHotEncoder(sparse=False)

        # prepare One Hot Encoding after kfold
        y_train = onehot_encoder.fit_transform(self.y_data[train].reshape((len(self.y_data[train]), 1)))
        y_test = onehot_encoder.fit_transform(self.y_data[test].reshape((len(self.y_data[test]), 1)))

        # defining model
        model = Sequential()
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(100, input_shape=(self.pre_length + 2 + self.post_length, 4), activation='sigmoid'))
        model.add(Dropout(0.5))

        model.add(Dense(40, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(2, activation='softmax'))

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
            model.summary(print_fn=lambda x: self.filehandler.write(x + '\n'))

            # print confusion matrix
            y_pred = model.predict(self.x_data[test])
            print("Confusion matrix:", confusion_matrix(y_true=y_test.argmax(axis=1), y_pred=y_pred.argmax(axis=1)),
                  file=self.filehandler)
            print("Confusion matrix:", confusion_matrix(y_true=y_test.argmax(axis=1), y_pred=y_pred.argmax(axis=1)))
            print("-----------------------------------------------------\n")