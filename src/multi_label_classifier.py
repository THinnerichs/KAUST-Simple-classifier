import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.callbacks import TensorBoard
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import confusion_matrix


def multi_label_classifier(x_data,
                      y_data,
                      filehandler,
                      cv_scores,
                      train,
                      test,
                      pre_length=300,
                      post_length=300):

    # parameters:
    epochs = 5
    batch_size = 500

    onehot_encoder = OneHotEncoder(sparse=False)

    # prepare One Hot Encoding after kfold
    y_train = onehot_encoder.fit_transform(y_data[train].reshape((len(y_data[train]), 1)))
    y_test = onehot_encoder.fit_transform(y_data[test].reshape((len(y_data[test]), 1)))

    # defining model
    model = Sequential()
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, input_shape=(pre_length + 2 + post_length, 4), activation='sigmoid'))
    model.add(Dropout(0.5))

    model.add(Dense(40, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))

    # compile model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # train model
    model.fit(x=x_data[train],
              y=y_train,
              epochs=epochs,
              batch_size=batch_size,
              callbacks=[TensorBoard(log_dir='/tmp/classifier')])

    model.summary()

    # evaluate the model
    scores = model.evaluate(x_data[test], y_test, verbose=0)

    print("\n--------------------------------------------------")
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    print("--------------------------------------------------\n")
    cv_scores.append(scores[1] * 100)

    if len(cv_scores) == 1:
        print("MULTI LABEL APPROACH", file=filehandler)
        print("Data shape: {}".format(x_data.shape), file=filehandler)
        model.summary(print_fn=lambda x: filehandler.write(x + '\n'))

        # print confusion matrix
        y_pred = model.predict(x_data[test])
        print("Confusion matrix:", confusion_matrix(y_true=y_test.argmax(axis=1), y_pred=y_pred.argmax(axis=1)), file=filehandler)
        print("Confusion matrix:", confusion_matrix(y_true=y_test.argmax(axis=1), y_pred=y_pred.argmax(axis=1)))
        print("-----------------------------------------------------\n")
