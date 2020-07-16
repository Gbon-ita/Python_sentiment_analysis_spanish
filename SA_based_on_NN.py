# -*- coding: utf-8 -*-
"""Sistema_de_Sentiment_Anlysis_based_on_Machine_Learning_with_Neural_Network

Created by Giovanni E. Bonaventura

"""
from my_emoji import*
import pandas as pd
import numpy as np
import re
import os
from collections import Counter

EMOJI=True

#!pip install stop_words

from stop_words import get_stop_words


training = pd.read_csv("training.txt",sep='\t',names=["id", "class", "tweet"])
dev = pd.read_csv("development.txt",sep='\t',names=["id", "class", "tweet"])

if EMOJI:
	training=encoding_emoji(training)
	dev=encoding_emoji(dev)

training['class']=pd.Categorical(training['class'], ['N','P','NEU','NONE'], ordered=True)
training=pd.get_dummies(training,columns=['class'])

dev['class']=pd.Categorical(dev['class'], ['N','P','NEU','NONE'], ordered=True)
dev['class']=dev['class'].cat.codes


"""# Toknenización"""

from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer(strip_handles=True,reduce_len=True)
list_words_tweets = []
for i in training.index:
  for w in tknzr.tokenize(training.loc[i,'tweet']):
    list_words_tweets.append(w.lower())
len(list_words_tweets)


stop_words = get_stop_words('es')
def count_dict(list_words_tweets):
  nonPunct = re.compile('.*[A-Za-z].*')  # must contain a letter or digit
  filtered = []
  for w in list_words_tweets:
    if nonPunct.match(w):
      if w not in stop_words:
        filtered.append(w)
  counts = Counter(filtered)
  counts=dict(counts)
  return counts

counts=count_dict(list_words_tweets)
"""# Toknenización con n-grams"""

from nltk import ngrams
def n_grams(df,n):
	list_n_grams_tweets=[]
	for i in df.index:
		string=df.loc[i,'tweet'].lower()
		list_n_grams_tweets.extend(ngrams(string.split(), n))
	return list_n_grams_tweets

list_2_grams_tweets=n_grams(training,2)
list_3_grams_tweets=n_grams(training,3)

list_2_grams=[]
for grams in list_2_grams_tweets:
 	list_2_grams.append(grams[0]+' '+grams[1])

list_3_grams=[]
for grams in list_3_grams_tweets:
 	list_3_grams.append(grams[0]+' '+grams[1]+' '+grams[2])


counts_2_grams=count_dict(list_2_grams)
counts_3_grams=count_dict(list_3_grams)



"""# Creaciòn bolsa de palabras"""

###numero minimo de ripeticiones### 
freq_words=1
freq_2_grams=100
freq_3_grams=100
###################################

words_for_bow=[]
for w in list(counts.keys()):
	if counts[w]>freq_words:
		words_for_bow.append(w)

words_for_bow_2_grams=[]
for w in list(counts_2_grams.keys()):
	if counts_2_grams[w]>freq_2_grams:
		words_for_bow_2_grams.append(w)

words_for_bow_3_grams=[]
for w in list(counts_3_grams.keys()):
	if counts_3_grams[w]>freq_3_grams:
		words_for_bow_3_grams.append(w)



list_of_bow=words_for_bow+words_for_bow_2_grams+words_for_bow_3_grams
print('number of variable: {}'.format(len(list_of_bow)))
print('number of words:{}'.format(len(words_for_bow)))
print('number of 2-grams:{}'.format(len(words_for_bow_2_grams)))
print('number of 3-grams:{}'.format(len(words_for_bow_3_grams)))



bow=pd.DataFrame(columns=list_of_bow,index=training.index)

bow.fillna(value=0,inplace=True)

for i in training.index:
  for w in tknzr.tokenize(training.loc[i,'tweet']):
    w=w.lower()
    if w in words_for_bow:
      bow.loc[i,w]+=1

for i in training.index:
	string=training.loc[i,'tweet'].lower()
	list_grams=[]
	for grams in ngrams(string.split(), 2):
		list_grams.append(grams[0]+' '+grams[1])
	for w in list_grams:  
		if w in words_for_bow_2_grams:
			bow.loc[i,w]+=1

for i in training.index:
	string=training.loc[i,'tweet'].lower()
	list_grams=[]
	for grams in ngrams(string.split(), 3):
		list_grams.append(grams[0]+' '+grams[1]+' '+grams[2])
	for w in list_grams:  
		if w in words_for_bow_3_grams:
			bow.loc[i,w]+=1

print(bow.head())

"""# Cojunto de test"""

bow_dev=pd.DataFrame(columns=list_of_bow,index=dev.index)
bow_dev.fillna(value=0,inplace=True)
for i in dev.index:
	for w in tknzr.tokenize(dev.loc[i,'tweet']):
		w=w.lower()
		if w in words_for_bow:
			bow_dev.loc[i,w]+=1

for i in dev.index:
	string=dev.loc[i,'tweet'].lower()
	list_grams=[]
	for grams in ngrams(string.split(), 2):
		list_grams.append(grams[0]+' '+grams[1])
	for w in list_grams:  
		if w in words_for_bow_2_grams:
			bow_dev.loc[i,w]+=1

for i in dev.index:
	string=dev.loc[i,'tweet'].lower()
	list_grams=[]
	for grams in ngrams(string.split(), 3):
		list_grams.append(grams[0]+' '+grams[1]+' '+grams[2])
	for w in list_grams:  
		if w in words_for_bow_3_grams:
			bow_dev.loc[i,w]+=1

"""# Creacion Modelo"""

from sklearn import svm

X_train=bow.values
y_train=training[['class_N','class_P','class_NEU','class_NONE']].values
X_test=bow_dev.values
y_test=dev['class'].values

X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)

##############SCALER##############
#scal=True
scal=False
if scal:
	from sklearn.preprocessing import StandardScaler

	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train=scaler.transform(X_train)
	X_test=scaler.transform(X_test)

#################################



#Neural Network
import keras

from keras.models import Sequential, Model, load_model
from keras.layers import Input, Reshape, Flatten, Activation
from keras.layers import Dropout, BatchNormalization, GaussianNoise
from keras.layers import Dense, LSTM
from keras.constraints import maxnorm
from keras.optimizers import RMSprop, SGD, Adadelta, Adagrad, Adam



###DEFINE###
_tensor_ = input_tensor = Input( shape=( X_train.shape[1], X_train.shape[2] ), name='main_input' )


_tensor_ = Flatten()(_tensor_)


_tensor_ = Dense( 200, activation='relu', kernel_constraint=maxnorm(3.0) )(_tensor_)
_tensor_ = Dense( 200, activation='relu', kernel_constraint=maxnorm(3.0) )(_tensor_)
_tensor_ = Dense( 200, activation='tanh', kernel_constraint=maxnorm(3.0) )(_tensor_)
_tensor_ = Dense( 100, activation='sigmoid', kernel_constraint=maxnorm(3.0) )(_tensor_)
_tensor_ = Dense( 50, activation='linear', kernel_constraint=maxnorm(3.0) )(_tensor_)


output_tensor = Dense( 4, activation='sigmoid', kernel_constraint=maxnorm(3.0) )(_tensor_)


model=Model( input_tensor, output_tensor )
model.compile( loss='mse', optimizer=Adagrad() ) 
model.summary()

###FIT###
model.fit( x_train, y_train, batch_size=300, epochs=150, shuffle=True, verbose=1 )


###PREDICT###

y_pred=model.predict(X_test)
y_pred_true=np.argmax(y_pred, axis=1)

accuracy=( 100.0 * (y_test == y_pred_true).sum() ) / len(y_test)
print('accuracy={}%'.format(accuracy))

var=input('do you want save this model? (y,n)')
if var=='y':
	import pickle
	pickle_out = open("modelo_NN",'wb')
	pickle.dump(model,pickle_out)
	pickle_out.close()

	pickle_out = open('scaler_NN','wb')
	pickle.dump(scaler,pickle_out)
	pickle_out.close()
