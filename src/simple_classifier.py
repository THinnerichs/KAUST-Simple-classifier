import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.callbacks import TensorBoard


def simple_classifier(x_data,
                      y_data,
                      filehandler,
                      cv_scores,
                      train,
                      test,
                      pre_length=300,
                      post_length=300):

    # parameters:
    epochs = 10
    batch_size = 500

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

    if len(cv_scores) == 10:
        print("BINARY CLASSIFICATION APPROACH", file=filehandler)
        print("Data shape: {}".format(x_data.shape), file=filehandler)
        model.summary(print_fn=lambda x: filehandler.write(x + '\n'))
