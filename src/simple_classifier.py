import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.callbacks import TensorBoard
from sklearn.model_selection import StratifiedKFold


def simple_classifier(load_file_name="acceptor",
                      results_log_file="../results/results_log",
                      samples_per_file=10000,
                      pre_length=300,
                      post_length=300):
    """
    This function applies the a simple binary classifier with the usage of Googles Keras onto the preprocessed data.
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
    x_data = np.load(file="../data/x_" + load_file_name + "_" + str(samples_per_file) + "_samples_" + str(pre_length) + "_pre_" + str(post_length) + "_post" + ".npy")
    y_data = np.load(file="../data/y_" + load_file_name + "_" + str(samples_per_file) + ".npy")
    print("Finished reading data in {}. x_data.shape {}, y_data.shape {}".format(time.time()-start,
                                                                                 x_data.shape,
                                                                                 y_data.shape))

    # Prepare train and test data
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    cv_scores = []

    # parameters:
    epochs = 10
    batch_size = 500

    # Perform Kfold cross validation
    for train, test in kfold.split(x_data, y_data):
        print("Round: {}".format(len(cv_scores) + 1))
        # defining model
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(200, input_shape=(pre_length + 2 + post_length, 4), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(80, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        # compile model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # train model
        model.fit(x=x_data[train],
                  y=y_data[train],
                  epochs=epochs,
                  batch_size=batch_size,
                  callbacks=[TensorBoard(log_dir='/tmp/classifier')])

        model.summary()

        # evaluate the model
        scores = model.evaluate(x_data[test], y_data[test], verbose=0)

        print("\n--------------------------------------------------")
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        print("--------------------------------------------------\n")
        cv_scores.append(scores[1] * 100)

        if len(cv_scores) == 9:
            with open(results_log_file, 'a') as fh:
                print("BINARY CLASSIFICATION APPROACH", file=fh)
                print("Data shape: {}".format(x_data.shape), file=fh)
                print("Mode:", load_file_name, file=fh)
                model.summary(print_fn=lambda x: fh.write(x + '\n'))

    print("Mean: {}, Std: {}".format(np.mean(cv_scores), np.std(cv_scores)))
    print("File name:", load_file_name)

    with open(file=results_log_file, mode='a') as fh:
        print("Mean: {}, Std: {}\n".format(np.mean(cv_scores), np.std(cv_scores)), file=fh)
        print("This took {} seconds.\n".format(time.time() - start), file=fh)
        print("\n-------------------------------------------------------------------------------\n", file=fh)


if __name__ == '__main__':
    test_start = time.time()
    simple_classifier(load_file_name="acceptor_data")
    simple_classifier(load_file_name="donor_data")
    print("This took {} seconds".format(time.time()-test_start))
