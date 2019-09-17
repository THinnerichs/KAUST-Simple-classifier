import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.callbacks import TensorBoard
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder


def multi_label_classifier(load_file_name="acceptor", results_log_file="../results/results_log"):
    """
    This function performs a multi label classification under usage of Googles Keras library.
    The model is validated with kfold cross validation implemented by the Scikit Learn library.
    Results are then logged to ../results/results_log
    
    :param load_file_name:          (string) The prefix of the saved .npy data file
    :param results_log_file:        (string) path to the logfile, where the results will be appended
    :return:                        None
    """
    start = time.time()
    seed = 12
    np.random.seed(seed)

    print("Reading data")
    x_data = np.load(file="../data/x_" + load_file_name + ".npy")
    y_data = np.load(file="../data/y_" + load_file_name + ".npy")
    print("Finished reading data in {}. x_data.shape {}, y_data.shape {}".format(time.time()-start,
                                                                                 x_data.shape,
                                                                                 y_data.shape))

    # Prepare train and test data
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    onehot_encoder = OneHotEncoder(sparse=False)

    cv_scores = []

    # parameters:
    epochs = 10
    batch_size = 500

    # Perform Kfold cross validation
    for train, test in kfold.split(x_data, y_data):
        print("Round: {}".format(len(cv_scores) + 1))

        # prepare One Hot Encoding after kfold
        y_train = onehot_encoder.fit_transform(y_data[train].reshape((len(y_data[train]), 1)))
        y_test = onehot_encoder.fit_transform(y_data[test].reshape((len(y_data[test]), 1)))

        # defining model
        model = Sequential()
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(150, input_shape=(602, 4), activation='relu'))
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

        if len(cv_scores) == 9:
            with open(results_log_file, 'a') as fh:
                print("MULTI LABEL APPROACH", file=fh)
                print("Data shape: {}".format(x_data.shape), file=fh)
                print("Mode:", load_file_name, file=fh)
                model.summary(print_fn=lambda x: fh.write(x + '\n'))

    print("Mean: {}, Std: {}".format(np.mean(cv_scores), np.std(cv_scores)))
    print("File name:", load_file_name)

    with open(file=results_log_file, mode='a') as fh:
        print("Mean: {}, Std: {}\n".format(np.mean(cv_scores), np.std(cv_scores)), file=fh)
        print("This took {} seconds.\n".format(time.time() - start), file=fh)
        print("Epochs: {}, Batch size: {}".format(epochs, batch_size), file=fh)
        print("\n-------------------------------------------------------------------------------\n", file=fh)


if __name__ == '__main__':
    test_start = time.time()
    # multi_label_classifier(load_file_name="both_data")
    multi_label_classifier(load_file_name="acceptor_data")
    multi_label_classifier(load_file_name="donor_data")
    print("This took {} seconds".format(time.time()-test_start))
