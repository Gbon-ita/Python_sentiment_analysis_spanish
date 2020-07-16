# -*- coding: utf-8 -*-
"""Sistema_de_Sentiment_Anlysis_based_on_Machine_Learning_with_Neural_Network

Created by Giovanni E. Bonaventura

"""

from my_emoji import*
import pickle
import pandas as pd
import numpy as np
from nltk import ngrams
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer(strip_handles=True,reduce_len=True)

EMOJI=True

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

model=load_file('model')
bows_list=load_file('bow_NN')

#########################################################################

################################ PREDICT ################################
test = pd.read_csv("test.txt",sep='\t',names=["id", "class", "tweet"])
if EMOJI:
	test=encoding_emoji(test)
predict=dict([])
for i in test.index:
	x=create_bow(test,i,bows_list)
	x=x.reshape(x.shape[0],x.shape[1],1)
	y=model.predict(x)
	y_pred=np.argmax(y, axis=1)
	tweet=test.loc[i,'tweet']
	if y_pred==0:
		predict[test.loc[i,'id']]='N'
	elif y_pred==1:
		predict[test.loc[i,'id']]='P'
	elif y_pred==2:
		predict[test.loc[i,'id']]='NEU'
	else:
		predict[test.loc[i,'id']]='NONE'

	print('{} text:"{}" class:"{}" '.format(i,tweet,predict[test.loc[i,'id']]))
#########################################################################

############################## WRITE OUTPUT #############################
f=open( 'OUTPUT_NN.txt', 'w' )
for id in list(predict.keys()):
	f.write( "%s\t" % id )
	f.write( "%s\n" % predict[id] )



