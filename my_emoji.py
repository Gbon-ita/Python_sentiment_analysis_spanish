#pip install emoji
import emoji
def encoding_emoji(df):
	for i in df.index:
		string=''
		for w in df.loc[i,'tweet'].split():
			W=emoji.demojize(w)
			string=string+W+' '
		df.loc[i,'tweet']=string
	return df
