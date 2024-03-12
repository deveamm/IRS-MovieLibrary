from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Embedding
from time import sleep
import numpy as np

X_train = []
y_train = []
X_test = []
y_test = []

trainFiles = open("cTrain.txt", "r").readlines()

for vector in trainFiles:
    valores = vector.strip().split()
    fila = [int(valor) for valor in valores]
    X_train.append(fila)
X_train = np.asarray(X_train).astype(np.float32)

trainResultsFiles = open("rTrain.txt", "r").readlines()

for vector in trainResultsFiles:
    valores = vector.strip().split()
    fila = [int(valor) for valor in valores]
    y_train.append(fila)
y_train = np.asarray(y_train).astype(np.float32)

testFiles = open("cTest.txt", "r").readlines()

for vector in testFiles:
    valores = vector.strip().split()
    fila = [int(valor) for valor in valores]
    X_test.append(fila)
X_test = np.asarray(X_test).astype(np.float32)

testResultsFiles = open("rTest.txt", "r").readlines()

for vector in testResultsFiles:
    valores = vector.strip().split()
    fila = [int(valor) for valor in valores]
    y_test.append(fila)
y_test = np.asarray(y_test).astype(np.float32)


'''print('Valor: {} - Type: {}'.format(fila,type(fila)))
sleep(1)
for x in range(len(fila)):
    print('ValorLIS: {} - Type: {}'.format(fila[x], type(fila[x])))'''

# Se crea el modelo
num_vocab_terms = 13858 #numero de palabras en el vocabulario final
num_neurons = 250
model = Sequential(name="perceptron")
model.add(Dense(units=num_vocab_terms, activation='relu', input_dim=num_vocab_terms))
model.add(Dense(units=num_neurons, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Fit the model
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128, verbose=2)
model.fit(X_train, y_train, epochs=5, batch_size=128, verbose=2)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))