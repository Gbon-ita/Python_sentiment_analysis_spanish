# -*- coding: utf-8 -*-
"""Sistema_de_Sentiment_Anlysis_based_on_Machine_Learning_with_Support_Vector_Machine

Created by Giovanni E. Bonaventura

NOTE:
	This code performs the prediction in the order  NEU -> P -> N -> NONE
	beacouse in this case it is the order that maximizes accuracy.
	If you want to classify with another order, you need to change it.
"""

from my_emoji import*
import pickle
import pandas as pd
import numpy as np
from nltk import ngrams
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer(strip_handles=True,reduce_len=True)

EMOJI=False

def create_bow(df,i,list_of_bow):
	bow=pd.DataFrame(columns=list_of_bow[0]+list_of_bow[1]+list_of_bow[2],index=[0])

	bow.fillna(value=0,inplace=True)
	count=0
	for w in tknzr.tokenize(df.loc[i,'tweet']):
		w=w.lower()
		if w in list_of_bow[0]:
			bow.loc[0,w]+=1
			count+=1
	string=df.loc[i,'tweet'].lower()
	list_grams=[]
	for grams in ngrams(string.split(), 2):
		list_grams.append(grams[0]+' '+grams[1])
	for w in list_grams:  
		if w in list_of_bow[1]:
			bow.loc[0,w]+=1
			count+=1
	string=df.loc[i,'tweet'].lower()
	list_grams=[]
	for grams in ngrams(string.split(), 3):
		list_grams.append(grams[0]+' '+grams[1]+' '+grams[2])
	for w in list_grams:  
		if w in list_of_bow[2]:
			bow.loc[0,w]+=1
			count+=1
	print('this prediction is based on {} variables'.format(count))
	x=bow.values
	return x

def load_file(name):
	pickle_in = open("{}".format(name),'rb')
	var = pickle.load(pickle_in)
	return var

############################### LOAD DATA ###############################
model_list=[]
scaler_list=[]
bows_list=[]
for i in range(0,3):
	model_list.append(load_file('modelo_svm_class_{}'.format(i)))
	scaler_list.append(load_file('scaler_{}'.format(i)))
	bows_list.append(load_file('bow_{}'.format(i)))
#########################################################################

################################ PREDICT ################################
test = pd.read_csv("test.txt",sep='\t',names=["id", "class", "tweet"])
if EMOJI:
	test=encoding_emoji(test)
predict=dict([])
for i in test.index:
	x=create_bow(test,i,bows_list[2])
	x=scaler_list[2].transform(x)
	y=model_list[2].predict(x)
	if y==1:
		predict[test.loc[i,'id']]='NEU'
	else:
		x=create_bow(test,i,bows_list[1])
		x=scaler_list[1].transform(x)
		y=model_list[1].predict(x)
		if y==1:
			predict[test.loc[i,'id']]='P'
		else:
			x=create_bow(test,i,bows_list[0])
			x=scaler_list[0].transform(x)
			y=model_list[0].predict(x)
			if y==1:
				predict[test.loc[i,'id']]='N'
			else:
				predict[test.loc[i,'id']]='NONE'
	tweet=test.loc[i,'tweet']
	print('{} text:"{}" class:"{}" '.format(i,tweet,predict[test.loc[i,'id']]))
#########################################################################

############################## WRITE OUTPUT #############################
f=open( 'OUTPUT_SVM.txt', 'w' )
for id in list(predict.keys()):
	f.write( "%s\t" % id )
	f.write( "%s\n" % predict[id] )



