# -*- coding: utf-8 -*-
"""Sistema_de_Sentiment_Anlysis_based_on_Machine_Learning_with_Support_Vector_Machine

Created by Giovanni E. Bonaventura

"""
from my_emoji import*
import pandas as pd
import numpy as np
import re
import os
from collections import Counter

EMOJI=False

#!pip install stop_words

from stop_words import get_stop_words


training = pd.read_csv("training.txt",sep='\t',names=["id", "class", "tweet"])
dev = pd.read_csv("development.txt",sep='\t',names=["id", "class", "tweet"])

if EMOJI:
	training=encoding_emoji(training)
	dev=encoding_emoji(dev)

training['class']=pd.Categorical(training['class'], ['N','P','NEU','NONE'], ordered=True)
training['class']=training['class'].cat.codes

dev['class']=pd.Categorical(dev['class'], ['N','P','NEU','NONE'], ordered=True)
dev['class']=dev['class'].cat.codes

training.head()

dev.head()

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
freq_2_grams=7
freq_3_grams=4
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
y_train=training['class'].values
X_test=bow_dev.values
y_test=dev['class'].values


##############SCALER##############
scal=True
#scal=False
if scal:
	from sklearn.preprocessing import StandardScaler

	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train=scaler.transform(X_train)
	X_test=scaler.transform(X_test)

#################################



###########################
### 0=N 1=P 2=NEU 3=NONE###
one_vs_all=0
###########################
y_train[y_train==one_vs_all]=4
y_train[y_train!=4]=0
y_train[y_train==4]=1
y_test[y_test==one_vs_all]=4
y_test[y_test!=4]=0
y_test[y_test==4]=1
###########################

#SVM_GRID
best_kernel=0
best_degree=0
best_gamma=0
best_C=0
best_accuracy=0
##
kernels = [ 'rbf', 'sigmoid','poly']
degrees = [ 1, 2, 3, 4, 5 ]
gammas = [ 0.0001, 0.001, 0.01, 0.1, 1.0, 2.0 ]
values_of_C = [ 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 1.0e+1, 1.0e+2, 1.0e+3 ]
coef0 = 1.0

print( "Testing different combinations of kernels and values of different parameters ... " )
for kernel in kernels:
    if kernel == 'poly' :
        _degrees = degrees
    else:
        _degrees = [1]
    for degree in _degrees:
        for gamma in gammas:
            for C in values_of_C:
                classifier = svm.SVC( kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, C=C, max_iter=100, verbose=False )
                classifier.fit( X_train, y_train )
                y_pred = classifier.predict( X_test )

                accuracy = ( 100.0 * (y_test == y_pred).sum() ) / len(y_test)

                if accuracy > best_accuracy:
                    best_accuracy=accuracy
                    best_kernel=kernel
                    best_gamma=gamma
                    best_C=C
                    best_degree=degree 
                print( " %-7s  degree %3d  gamma %.6f  C %e  Accuracy %.1f%%" % (kernel, degree, gamma, C, accuracy ) )
print( "the best combination is: %-7s  degree %3d  gamma %.6f  C %e  Accuracy %.1f%%" % (best_kernel,best_degree, best_gamma, best_C, best_accuracy ) )


var=input('do you want save this model? (y,n)')
if var=='y':
	classifier_final = svm.SVC( kernel=best_kernel, degree=best_degree, gamma=best_gamma, coef0=coef0, C=best_C, max_iter=100, verbose=False )
	classifier_final.fit( X_train, y_train )
	import pickle
	pickle_out = open("modelo_svm_class_{}".format(one_vs_all),'wb')
	pickle.dump(classifier_final,pickle_out)
	pickle_out.close()

	pickle_out = open('scaler_{}'.format(one_vs_all),'wb')
	pickle.dump(scaler,pickle_out)
	pickle_out.close()

	pickle_out = open('bow_{}'.format(one_vs_all),'wb')
	pickle.dump([words_for_bow, words_for_bow_2_grams, words_for_bow_3_grams],pickle_out)
	pickle_out.close()

##########Results##########
'''
NONE vs ALL accuracy: 87.7%
NEU  vs ALL accuracy: 86.8%
POS  vs ALL accuracy: 72.7%
NEG  vs ALL accuracy: 62.6%
'''
###########################
