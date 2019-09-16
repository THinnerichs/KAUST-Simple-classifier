## Results of the performed models

### Dataset:

- The following models are performed on a data size of 10000 per file, equaling 20000 records all in all.
#### Test #1:
Model:
```python
model = Sequential()
model.add(Flatten())
model.add(Dense(602, input_shape=(602,4), activation='relu'))
model.add(Dense(80, activation='sigmoid'))
model.add(Dense(30, activation='tanh'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```
Epochs: 15
Batch size: 500

##### Acceptor
Mean: 91.22

Std: 0.5971599450733426

This took 274.82445645332336 seconds

##### Donor
Mean: 91.04499999999999

Std: 0.6908147363801678

This took 373.7056896686554 seconds

#### Test #2
Model:
```python
model = Sequential()
model.add(Flatten())
model.add(Dense(602, input_shape=(602,4), activation='relu'))
model.add(Dense(30, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```
Epochs: 15
Batch size: 500
##### Acceptor
Mean: 91.27

Std: 0.7909487973314062

This took 452.9686803817749 seconds

##### Donor
Mean: 91.27

Std: 0.7909487973314062

This took 452.9686803817749 seconds

#### Test #3
```python
model = Sequential()
model.add(Flatten())
model.add(Dense(602, input_shape=(602,4), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(80, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(30, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

Epochs: 10
Batch size: 500

##### Acceptor
Mean: 91.59500000000001

Std: 0.7315907325820914

This took 321.77250838279724 seconds

##### Donor
Mean: 91.355

Std: 0.798576859168862

This took 279.0536549091339 seconds


- The following models are performed on a data size of 20000 per file, equaling 40000 records all in all.

#### Test #4

```python
model.add(Flatten())
model.add(Dense(602, input_shape=(602,4), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(30, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

Epochs: 10
Batch size: 500

##### Acceptor
Mean: 92.25500000000001

Std: 0.6935236117105187

This took 441.35150027275085 seconds

##### Donor
Mean: 93.57000000000001

Std: 0.4514975082987719

This took 442.19666624069214 seconds
