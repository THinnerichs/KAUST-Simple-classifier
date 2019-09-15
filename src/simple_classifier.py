import numpy as np
from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import itertools

# Prepare and read data
print("Reading data ...")
label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False)

dataset = no
for a,b in itertools.product(["negative", "positive"], ["acceptor", "donor"]):
    # Read data
    file_name = "../data/{}_{}.fa".format(a,b)
    print("Processing", file_name)

    for record in SeqIO.parse(file_name, "fasta"):
        loop_record = np.array(record.seq, np.character)
        onehot_encoded = onehot_encoder.fit_transform(loop_record.reshape((len(loop_record), 1)))

    # Prepare labels




"""
x_dataset = []
for a,b in itertools.product(["negative", "positive"], ["acceptor", "donor"]):
     # Read data
    file_name = "../data/{}_DNA_seqs_{}_at.fa".format(a,b)
    print("Processing", file_name)
    counter = 0
    with open(file=file_name, mode="r") as f:
        for line in f:
            counter += 1
            if counter % 20000 == 0:
                print("Counter:", counter)

            record = []
            for character in line.strip():
                record.append(character)
            record = np.array(record)
            onehot_encoded = onehot_encoder.fit_transform(record.reshape((len(record),1)))

            x_dataset.append(onehot_encoded)

    # Prepare labels
    
x_dataset = np.array(x_dataset)

print("Finished reading data. Shape:", x_dataset.shape)
"""

# Prepare labels


# for record in SeqIO.parse("../data/negative_DNA_seqs_donor_at.fa", "fasta"):
#    print(record)

# align_array = np.array([list(rec) for rec in alignment], np.character)

# print(align_array)