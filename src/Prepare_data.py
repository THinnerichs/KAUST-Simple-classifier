import numpy as np
from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import itertools
import time


# parameters:
samples_per_file = 10000

start = time.time()
# Prepare and read data
print("Reading data ...")
label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False)

x_dataset = []
y_dataset = []
for a,b in itertools.product(["negative", "positive"], ["acceptor"]):
    # Read data
    file_name = "../data/{}_{}.fa".format(a,b)
    print("Processing", file_name)
    my_time = time.time()
    counter = 0

    for record in SeqIO.parse(file_name, "fasta"):
        loop_record = np.array(record.seq, np.character)
        onehot_encoded = onehot_encoder.fit_transform(loop_record.reshape((len(loop_record), 1)))

        x_dataset.append(onehot_encoded)
        counter += 1

        if counter % 2000 == 0:
            print("Processed records", counter, time.time() - my_time)
            my_time = time.time()
        if counter >= samples_per_file:
            break

    # Prepare y labels
    y_dataset.extend(counter * [a+b])
y_dataset = np.array(y_dataset)
y_dataset = label_encoder.fit_transform(y_dataset)
print("y_dataset shape:", y_dataset.shape)

x_dataset = np.array(x_dataset, dtype=np.int64)
print("x_dataset shape:", x_dataset.shape)
print("Finished reading data")

np.save(file="../data/x_dataset.npy", arr=x_dataset)
np.save(file="../data/y_dataset.npy", arr=y_dataset)

# data = np.load(file="../data/dataset.npy")
# print("Shape:", data.shape)

end = time.time()
print("This took {} seconds.".format(end-start))
