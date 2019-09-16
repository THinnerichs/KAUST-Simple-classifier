## Results of the performed models

### Dataset:

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
acc: 90.10%

Mean: 91.22

Std: 0.5971599450733426

This took 274.82445645332336 seconds

##### Donor
acc: 90.00%

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
acc: 89.95%

Mean: 91.27

Std: 0.7909487973314062

This took 452.9686803817749 seconds

##### Donor
acc: 89.95%

Mean: 91.27

Std: 0.7909487973314062

This took 452.9686803817749 seconds


