import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from sklearn.model_selection import StratifiedKFold


def simple_classifier(load_file_name="acceptor", results_log_file="../results/results_log"):
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

    cv_scores = []

    # parameters:
    epochs = 10
    batch_size = 500

    for train, test in kfold.split(x_data, y_data):
        print("Round: {}".format(len(cv_scores) + 1))
        # defining model
        model = Sequential()
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(100, input_shape=(602,4), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(80, activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(80, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        # compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # train model
        model.fit(x=x_data[train], y=y_data[train], epochs=epochs, batch_size=batch_size)

        model.summary()

        # evaluate the model
        scores = model.evaluate(x_data[test], y_data[test], verbose=0)

        print("\n--------------------------------------------------")
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        print("--------------------------------------------------\n")
        cv_scores.append(scores[1] * 100)

        if len(cv_scores) == 1:
            with open(results_log_file, 'a') as fh:
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
    simple_classifier(load_file_name="acceptor_data_100000")
    simple_classifier(load_file_name="donor_data_100000")
    print("This took {} seconds".format(time.time()-test_start))



