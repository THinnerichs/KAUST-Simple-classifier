import numpy as np
import time
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Flatten
from sklearn.model_selection import StratifiedKFold


def simple_classifier(load_file_name="dataset"):
    start = time.time()
    seed = 12
    np.random.seed(seed)

    print("Reading data")
    x_data = np.load(file="../data/x_" + load_file_name + ".npy")
    y_data = np.load(file="../data/y_" + load_file_name + ".npy")
    print("Finished reading data in {}. x_data.shape {}, y_data.shape {}".format(time.time()-start, x_data.shape,y_data.shape))

    # Prepare train and test data
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    cv_scores = []

    # parameters:
    epochs = 15
    batch_size = 500

    for train, test in kfold.split(x_data, y_data):
        # defining model
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(602, input_shape=(602,4), activation='relu'))
        model.add(Dense(80, activation='sigmoid'))
        model.add(Dense(30, activation='tanh'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        # compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # train model
        model.fit(x=x_data[train], y=y_data[train], epochs=epochs, batch_size=batch_size)

        #evaluate the model
        scores = model.evaluate(x_data[test], y_data[test], verbose=0)

        print("\n--------------------------------------------------")
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        print("--------------------------------------------------\n")
        cv_scores.append(scores[1] * 100)

    print("Mean: {}, Std: {}".format(np.mean(cv_scores), np.std(cv_scores)))

if __name__ == '__main__':
    test_start = time.time()
    simple_classifier(load_file_name="acceptor_data")
    print("This took {} seconds".format(time.time()-test_start))



