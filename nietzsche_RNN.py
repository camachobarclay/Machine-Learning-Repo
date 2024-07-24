import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
"""
from tensorflow.keras import layers
from tensorflow.keras import utils
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional
from keras.layers import SimpleRNN, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv1D, MaxPooling1D, ZeroPadding1D
from keras.utils import np_utils
from keras.optimizers import Adam
import cPickle as pickle
import bcolz
import re
from numpy.random import random, permutation, randn, normal, uniform, choice


"""

text = open('nietzsche.txt',"r")
nietzsche = text.read() 
text.close()
text2 = nietzsche

type2 = type(text2)


text = open('words_250000_train.txt',"r")
full_dictionary = text.read().splitlines()
text.close()
current_dictionary = []
for dict_word in full_dictionary:
    if len(dict_word)!=7:
        continue
    else:
        current_dictionary.append(dict_word)

text = ' '.join(full_dictionary)

type1 = type(text)


#print("length of the text = ",len(text))
print("\n\n"); print(text); print("\n\n")


print("\n\n")
print("type of dictionary is :",type1)
#print("type of dictionary is :",type2)
print("\n\n")

chars = sorted(list(set(text)))
print("length of chars + 1 =  ",len(chars)+1)

#chars.insert(0, '\0')

char_to_index = {v:i for i,v in enumerate(chars)}
index_to_char = {i:v for i,v in enumerate(chars)}

total_index = [char_to_index[char] for char in text]

print("total index[:10] = ",total_index[:10])

print("''.join(index_to_char[i] for i in total_index[:25]) = ", ''.join(index_to_char[i] for i in total_index[:25]))

pred_num = 6
xin = [[total_index[j+i] for j in range(0, len(total_index)-1-pred_num, pred_num)] for i in range(pred_num)]
y = [total_index[i+pred_num] for i in range(0, len(total_index)-1-pred_num, pred_num)]

X = [np.stack(xin[i][:-2]) for i in range(pred_num)]
Y = np.stack(y[:-2])

print("X =", X)


print("Y[:8] = ", Y[:8])

print("X[0] shape",X[0].shape)
print("Y.shape", Y.shape)


hidden_layers = 256
vocab_size = len(chars)
n_fac = 42

model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, n_fac, input_length=pred_num),
        keras.layers.SimpleRNN(hidden_layers, activation='relu'),
        keras.layers.Dense(vocab_size, activation='sigmoid')
    ])

model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam())
model.fit(np.stack(X, 1), Y, batch_size=20, epochs=5)
"""
model.save_weights('simpleRNN_3pred.h5')
model.load_weights('simpleRNN_3pred.h5')
model.save_weights('simpleRNN_7pred.h5')
model.load_weights('simpleRNN_7pred.h5')
"""
def predict_next_char(inp):
    index = [char_to_index[i] for i in inp]
    arr = np.expand_dims(np.array(index), axis=0)
    prediction = model.predict(arr)
    print("Printing prediction for ",inp)
    print(index_to_char[np.argmax(prediction)])

predict_next_char('allerg')
predict_next_char('generi')
predict_next_char('specia')
predict_next_char('dietar')

"""
ys = [[total_index[j+i] for j in range(1, len(total_index)-pred_num, pred_num)] for i in range(pred_num)]
Y_return = [np.stack(ys[i][:-2]) for i in range(pred_num)]
print("X = ",X)
print("Y_return",Y_return)
vocab_size = 86
n_fac = 42
hidden_layers = 256

return_model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, n_fac, input_length=pred_num),
        tf.keras.layers.SimpleRNN(hidden_layers, return_sequences=True, activation='relu'),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size, activation='softmax'))
    ])

return_model.summary()
return_model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam())

X_model = np.stack(X, 1)
Y_model = np.expand_dims(np.stack(Y_return, 1), axis=-1)

return_model.fit(X_model, Y_model, batch_size=64, epochs=5)

return_model.optimizer.lr = 1e-4
return_model.fit(X_model, Y_model, batch_size=64, epochs=5)

return_model.optimizer.lr = 1e-4
return_model.fit(X_model, Y_model, batch_size=64, epochs=5)

return_model.save_weights('return_sequences_25.h5')

def predict_every_char(inp):
    l = []
    p = 0
    while p<len(inp):
        pre_inp = inp[p:p+pred_num]
        if len(pre_inp) < pred_num:
            pre_inp = pre_inp + ' '*(pred_num - len(pre_inp))
            l.append(pre_inp)
        else:
            l.append(pre_inp) 
        p+=pred_num
#     index = [char_to_index[i] for i in inp]
#     arr = np.expand_dims(index, axis=0)
#     prediction = return_model.predict(arr)
#     return ''.join([index_to_char[np.argmax(i)] for i in prediction[0]])    
    final = []
    for half in l:
        index = [char_to_index[i] for i in half]
        arr = np.expand_dims(index, axis=0)
        prediction = return_model.predict(arr)
        final.append(''.join([index_to_char[np.argmax(i)] for i in prediction[0]]))
    return ''.join(final)

a = predict_every_char('and the boy left')   
b = predict_every_char('this is')
c = predict_every_char("140 After having discovered in many of the less comprehensible actions mere manifestations of pleasure in emotion for its own sake, I fancy I can detect in the self contempt which characterises holy persons, and also in their acts of self torture (through hunger and scourgings, distortions and chaining of the limbs, acts of madness) simply a means whereby such natures may resist the general exhaustion of their will to live (their nerves). They employ the most painful expedients to escape if only for a time from the heaviness and weariness in which they are steeped by their great mental indolence and their subjection to a will other than their own.")

print("a string = ", a)
print("b string = ", b)
print("c string = ", c)
"""