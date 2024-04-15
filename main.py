import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from art import text2art

ascii_art = text2art("RouletteAi")

print(ascii_art)
print("Roulette prediction artificial intelligence")

data = np.genfromtxt('data.txt', delimiter='\n', dtype=int)

data = data[(data >= 0) & (data <= 36)]

sequence_length = 10

sequences = np.array([data[i:i+sequence_length] for i in range(len(data)-sequence_length)])

targets = data[sequence_length:]

train_data = sequences[:int(0.8*len(sequences))]
train_targets = targets[:int(0.8*len(targets))]
val_data = sequences[int(0.8*len(sequences)):]
val_targets = targets[int(0.8*len(targets)):]

max_value = np.max(data)

num_features = 1

model = keras.Sequential()
model.add(layers.Embedding(input_dim=max_value+1, output_dim=64))
model.add(layers.LSTM(256))
model.add(layers.Dense(num_features, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


history = model.fit(train_data, train_targets, validation_data=(val_data, val_targets), epochs=100)

predictions = model.predict(val_data)

indices = np.argsort(predictions, axis=1)[:, -num_features:]
predicted_numbers = np.take_along_axis(val_data, indices, axis=1)

print("============================================================")
print("Predicted Number:")
for numbers in predicted_numbers[:1]:
    print(', '.join(map(str, numbers)))

input('Press ENTER to exit')