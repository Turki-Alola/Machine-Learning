#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 14:47:10 2018

@author: turki
"""

def word2int64(word):
    n64 = 0;
        # print("length =",length(word))
    for ch in word:
        #print(ord(ch)-1569)
        k=ord(ch)-1569
        n64=n64*64+k
    return(n64)


def logbase64(n):
    r = round(math.log2(n)/math.log2(64))
    return r

def RecoverString(num64):
    RetS =''
    N =logbase64(num64)-1
    #print("N=",N)
    snum =num64
    while N>=0 :
         posn = snum // 64**N
         #print("          posn=", posn)
         RetS = RetS + chr(posn+1569)
         snum = snum % 64**N
         N=N-1
    return(RetS)






#Import needed libraries and perform initializations
import numpy
import numpy as np
import pandas
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.utils import to_categorical

from numpy import array
from numpy import argmax
from sklearn.preprocessing import OneHotEncoder

from keras.preprocessing.text import one_hot
from keras.preprocessing.text import text_to_word_sequence
import codecs
import csv
import math

import seaborn as sns
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

seed = 7
numpy.random.seed(seed)



#Step 2: RIT64 Encoding/Decoding Algorithms

def word2int64(word):
    n64 = 0;
        # print("length =",length(word))
    for ch in word:
        #print(ord(ch)-1569)
        k=ord(ch)-1569
        n64=n64*64+k
    return(n64)


def logbase64(n):
    r = round(math.log2(n)/math.log2(64))
    return r

def RecoverString(num64):
    RetS =''
    N =logbase64(num64)-1
    #print("N=",N)
    snum =num64
    while N>=0 :
         posn = snum // 64**N
         #print("          posn=", posn)
         RetS = RetS + chr(posn+1569)
         snum = snum % 64**N
         N=N-1
    return(RetS)

#Step 3:
#Load the data, perform encoding for the input layer and create initial data lists
wordList=[]
posList=[]
maxw=0
maxp=0
XXX = [] # List of Vectors for the input Layer 
with open('uq4.csv', encoding='utf-8') as csv_file:# csv standard format for text reading
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        word =row[0] # the word
        pos = row[1] # the tag
        print("word=", word, "    pos=", pos)
        wint64 = word2int64(word)  # RIT64 encoding for words
        wbin = bin(wint64)         # binary format 
        pint64= word2int64(pos)    #RIT64 for POS tags
        pbin = bin(pint64)         #binary format 
       
        wordList.append(word)      # create the word list
        posList.append(pos)        # create the pos tag list
        
        #convert RIT64 encoding into a list of binary digits for neurons
        wbinList = [int(x) for x in bin(wint64)[2:]]  # create binary vector for word
        print("wbin= ", wbin)
        Lead = [0]*(60 -len(wbinList)) # add leading zero elements
        wbList =Lead +wbinList  # add leading zeros to input layer vector to make it 60
        XXX.append(wbList)
        print(wbList)
        
        
XAA = array(XXX)  # cast the XXX list into a NumPy array
print(XAA)

#Step 4:  One Hot Encoding
#One Hot encoding for the output layer – and the input layer (for later comparisons)
# Preparing for one hot encoding
values = array(wordList)
#integer encode
wrs_encoder = LabelEncoder()
wrs_integer_encoded = wrs_encoder.fit_transform(values)
wrs_onehot_encoder = OneHotEncoder(sparse=False)

wrs_integer_encoded = wrs_integer_encoded.reshape(len(wrs_integer_encoded), 1)
wrs_onehot_encoded = wrs_onehot_encoder.fit_transform(wrs_integer_encoded)


#inverted = wrs_encoder.inverse_transform([argmax(wrs_onehot_encoded[0, :])])
#print(inverted)
results = 'results:'
#OneHotFile.close()
values2 = array(posList)
# integer encode
pts_encoder = LabelEncoder()
pts_integer_encoded = pts_encoder.fit_transform(values2)

# binary encode
pts_onehot_encoder = OneHotEncoder(sparse=False)
pts_integer_encoded = pts_integer_encoded.reshape(len(pts_integer_encoded), 1)
pts_onehot_encoded = pts_onehot_encoder.fit_transform(pts_integer_encoded)

#one hot encoded input and target vectors: X for input, Y for Target
X=array(wrs_onehot_encoded)
Y=array(pts_onehot_encoded)

#check shapes
print("shape of Y=", Y.shape)
print("dimesiion of X=", X.shape)

print("length of XAA is ", len(XAA))
print("the shape of XA is ", XAA.shape[1])

#Step 5: Train and Test using RIT64 Version
XAtr, XAts, Ytr, Yts = train_test_split( XAA, Y, test_size=0.25, random_state=87)
numpy.random.seed(seed)
#first model
print('\nloading first model\n')
model = Sequential()
model.add(Dense(60, input_dim=60, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(88, activation='softmax'))
#second model
print('\nloading 2nd model\n')
model2 = Sequential()
model2.add(Dense(60, input_dim=60, activation='softmax'))
model2.add(Dense(256, activation='softmax'))
model2.add(Dense(256, activation='softmax'))
model2.add(Dense(128, activation='softmax'))
model2.add(Dense(88, activation='softmax'))

print('\nloading 3rd model\n')
model3 = Sequential()
model3.add(Dense(60, input_dim=60, activation='sigmoid'))
model3.add(Dense(256, activation='sigmoid'))
model3.add(Dense(256, activation='sigmoid'))
model3.add(Dense(128, activation='sigmoid'))
model3.add(Dense(88, activation='softmax'))


#adam
#---------------------------
#first round

# Compile model
print('COMPILING')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# Fit the model
t=time.process_time()
print('FITTING')
model.fit(XAtr, Ytr, epochs=20, batch_size=10)
scores = model.evaluate(XAts, Yts)
results+="\n model x adam\n %s: %.2f%%\n" % (model.metrics_names[1], scores[1]*100)

#second round
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model2.fit(XAtr, Ytr, epochs=20, batch_size=10)
scores = model2.evaluate(XAts, Yts)
results+="\n model2 x adam\n %s: %.2f%%\n" % (model2.metrics_names[1], scores[1]*100)
#third round
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model3.fit(XAtr, Ytr, epochs=20, batch_size=10)
scores = model3.evaluate(XAts, Yts)
results+="\n model3 x adam\n %s: %.2f%%\n" % (model3.metrics_names[1], scores[1]*100)
#sgd
#------------------------------
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])
# Fit the model
t=time.process_time()
print('FITTING')
model.fit(XAtr, Ytr, epochs=20, batch_size=10)
scores = model.evaluate(XAts, Yts)
results+="\n model x sgd\n %s: %.2f%%\n" % (model.metrics_names[1], scores[1]*100)

#second round
model2.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])
model2.fit(XAtr, Ytr, epochs=20, batch_size=10)
scores = model2.evaluate(XAts, Yts)
results+="\n model2 x sgd\n %s: %.2f%%\n" % (model2.metrics_names[1], scores[1]*100)
#third round
model3.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])
model3.fit(XAtr, Ytr, epochs=20, batch_size=10)
scores = model3.evaluate(XAts, Yts)
results+="\n model3 x sgd\n %s: %.2f%%\n" % (model3.metrics_names[1], scores[1]*100)
#adamax
#------------------------------
model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['acc'])
# Fit the model
t=time.process_time()
print('FITTING')
model.fit(XAtr, Ytr, epochs=20, batch_size=10)
scores = model.evaluate(XAts, Yts)
results+="\n model x adamax\n %s: %.2f%%\n" % (model.metrics_names[1], scores[1]*100)

#second round
model2.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['acc'])
model2.fit(XAtr, Ytr, epochs=20, batch_size=10)
scores = model2.evaluate(XAts, Yts)
results+="\n model2 x adamax\n %s: %.2f%%\n" % (model2.metrics_names[1], scores[1]*100)
#third round
model3.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['acc'])
model3.fit(XAtr, Ytr, epochs=20, batch_size=10)
scores = model3.evaluate(XAts, Yts)
results+="\n model3 x adama\n %s: %.2f%%\n" % (model3.metrics_names[1], scores[1]*100)
print(results)
'''
# evaluate the model
print('EVALUATING')
scores = model.evaluate(XAts, Yts)
print("\n%s: %.2f%%\n" % (model.metrics_names[1], scores[1]*100))
scores = model2.evaluate(XAts, Yts)
print("\n%s: %.2f%%\n" % (model2.metrics_names[1], scores[1]*100))
elapsed_time=time.process_time() - t
print("elapsed time is =", elapsed_time)
print('DONE EVALUATING')
'''

#Step 6: Train and test Using One Hot Encoding 
#(Warning: make take more that hour to finish -Try at home execution disabled in the code)
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.25, random_state=87)
#print("shape of X Train =", X_train.shape)
numpy.random.seed(seed)
model = Sequential() # create model
model.add(Dense(units=7509, input_shape=(7509,), activation='relu')) # hidden layer
model.add(Dense(units=88, activation='sigmoid')) # output layer
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

t=time.process_time()
#model.fit(X_train, Y_train, epochs=1, verbose=0)

#scores = model.evaluate(X_test, Y_test)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#elapsed_time=time.process_time() - t
#print("elapsed time is =", elapsed_time)

#How to Save and Reload the Model
#1-save model
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# Later ...... Later
#2- Load Saved Model
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


 
# 3-evaluate loaded model on test data
#loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
