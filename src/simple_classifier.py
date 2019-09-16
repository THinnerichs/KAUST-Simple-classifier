import numpy as np
import time
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Flatten


start = time.time()

print("Reading data")
x_data = np.load(file="../data/x_dataset.npy")
y_data = np.load(file="../data/y_dataset.npy")
print("Finished reading data in {}. x_data.shape {}, y_data.shape {}".format(time.time()-start, x_data.shape,y_data.shape))

def simple_classifier():

    # parameters:
    epochs = 50
    batch_size = 50

    # defining model
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(602, input_shape=(602,4), activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(80, activation='sigmoid'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(30, activation='softmax'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("BUMM")
    # train model
    model.fit(x=x_data, y=y_data, validation_split=0.33, epochs=epochs, batch_size=batch_size)

if __name__ == '__main__':
    test_start = time.time()
    simple_classifier()
    print("This took {} seconds".format(time.time()-test_start))



