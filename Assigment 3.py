#Import needed libraries and perform initializations
import numpy
import numpy as np
import pandas
import time
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.utils import to_categorical
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

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
x=0
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
    
        
#cast the XXX list into a NumPy array    
XAA = array(XXX)  
print(XAA)
def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]

#split the corpus
XAA1, XAA2 = split_list(XAA)
wordList1, wordList2 = split_list(wordList)
posList1, posList2 = split_list(posList)

#print("wordList1: ", len(wordList1))
#print("posList1: ", len(posList1))
#print("wordList2: ", len(wordList2))
#print("posList2: ", len(posList2))
#print("wordList: ", len(wordList))
#print("posList: ", len(posList))

results = "results:\n"

#wordList, posList: words and their position
#thing: for the output layer since you need mutiple different values for each
#XAA: the binary list associated with each
#tag: added to the results string, so we can differenciate between each
def train (wordList,posList, thing, XAA, activationFunction):
    global results
    values = array(wordList)
    #integer encode
    wrs_encoder = LabelEncoder()
    wrs_integer_encoded = wrs_encoder.fit_transform(values)
    #wrs_onehot_encoder = OneHotEncoder(sparse=False)
    
    wrs_integer_encoded = wrs_integer_encoded.reshape(len(wrs_integer_encoded), 1)
    #wrs_onehot_encoded = wrs_onehot_encoder.fit_transform(wrs_integer_encoded)

    #OneHotFile.close()
    values2 = array(posList)
    #integer encode
    pts_encoder = LabelEncoder()
    pts_integer_encoded = pts_encoder.fit_transform(values2)
    
    #binary encode
    pts_onehot_encoder = OneHotEncoder(sparse=False)
    pts_integer_encoded = pts_integer_encoded.reshape(len(pts_integer_encoded), 1)
    pts_onehot_encoded = pts_onehot_encoder.fit_transform(pts_integer_encoded)
    
    #x1 = array(wrs_onehot_encoded)
    y1 = array(pts_onehot_encoded)
    numpy.random.seed(seed)
    model = Sequential()
    model.add(Embedding(2, 32, input_length=60))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(thing, activation = activationFunction))
    XAtr, XAts, Ytr, Yts = train_test_split( XAA, y1, test_size=0.25, random_state=87)
    
    #binary_crossentropy
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(XAtr, Ytr, epochs=5, batch_size=10)
    scores = model.evaluate(XAts, Yts)
    results+= "%s with binary_crossentropy %s: %.2f%%\n" % (activationFunction, model.metrics_names[1], scores[1]*100)
    
    #mean_squared_error
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(XAtr, Ytr, epochs=5, batch_size=10)
    scores = model.evaluate(XAts, Yts)
    results+= "%s with mean_squared_error %s: %.2f%%\n" % (activationFunction, model.metrics_names[1], scores[1]*100)
    
    #mean_absolute_error
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    model.fit(XAtr, Ytr, epochs=5, batch_size=10)
    scores = model.evaluate(XAts, Yts)
    results+= "%s with mean_absolute_error %s: %.2f%%\n" % (activationFunction, model.metrics_names[1], scores[1]*100)
    
    #mean_squared_logarithmic_error
    model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=['accuracy'])
    model.fit(XAtr, Ytr, epochs=5, batch_size=10)
    scores = model.evaluate(XAts, Yts)
    results+= "%s with mean_squared_logarithmic_error %s: %.2f%%\n" % (activationFunction, model.metrics_names[1], scores[1]*100)
    
    #categorical_crossentropy
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(XAtr, Ytr, epochs=5, batch_size=10)
    scores = model.evaluate(XAts, Yts)
    results+= "%s with categorical_crossentropy %s: %.2f%%\n" % (activationFunction, model.metrics_names[1], scores[1]*100)
    
    #cosine_proximity
    model.compile(loss='cosine_proximity', optimizer='adam', metrics=['accuracy'])
    model.fit(XAtr, Ytr, epochs=5, batch_size=10)
    scores = model.evaluate(XAts, Yts)
    results+= "%s with cosine_proximity %s: %.2f%%\n" % (activationFunction, model.metrics_names[1], scores[1]*100)
    
    


train(wordList,posList,88, XAA, "sigmoid")
train(wordList,posList,88, XAA, "relu")
train(wordList,posList,88, XAA, "softmax")
train(wordList,posList,88, XAA, "tanh")
print(results)