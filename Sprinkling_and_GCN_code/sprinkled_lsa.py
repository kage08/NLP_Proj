import nltk
from nltk.corpus import wordnet as wn
import re
from gensim import models, corpora
from nltk import word_tokenize
import pandas as pd
import numpy as np
from nltk.corpus import wordnet_ic, brown, gutenberg, pros_cons, movie_reviews, product_reviews_1, product_reviews_2, genesis, reuters,webtext,inaugural,stopwords
import scipy as sp
import scipy.stats
from argparse import ArgumentParser
import math
from sklearn.preprocessing import normalize
import sklearn
import sklearn.decomposition
import cPickle as pickle
import os
from cython.parallel import prange
from gensim.matutils import corpus2csc

parser = ArgumentParser()
parser.add_argument('--similarity_function',choices=['path','lch','wup','res','jcn','lin'],default='path', type=str,help='type of similarity')
parser.add_argument('--pooling',choices=['average','max','min','top'],default='max', type=str,help='pooling type. In case of top, it is only valid for path and jcn')

parser.add_argument('--dataset',choices=['brown', 'gutenberg', 'pros_cons', 'movie_reviews', 'product_reviews_1', 'product_reviews_2', 'genesis', 'reuters','webtext','inaugural'],default='brown', type=str,help='Corpus')
parser.add_argument('--latent_dim',default=100, type=int,help='Size of embedding')
parser.add_argument("--stretch", action="store_true", help='If used, stretching by singular values done')
parser.add_argument("--l2_normalize", action="store_true", help='If used, l2 - normalization of embeddings used ')
args = parser.parse_args()

def convert_none_to_zero(score_list):
	if(score_list==[]):
		return [0]
	else:	
		return	[0 if v is None else min(v,100) for v in score_list]

def get_pos_synset(synset):
	name = synset.name()
	lemma, pos, synset_index_str = name.lower().rsplit('.', 2)
	
	return pos


def get_score(word_1,word_2,pooling,same_pos_comparison=False):
	if(args.similarity_function not in ['path','wup']):
		same_pos_comparison=True
	if(args.similarity_function  in ['res','jcn','lin']):
		information_based_bool=True
	else:
		information_based_bool = False			
	synset_1_list = wn.synsets(word_1)
	synset_2_list = wn.synsets(word_2)
	if(synset_1_list==[] or synset_2_list==[]):
		return 0

		
	
	else:
		all_scores = []

		for synset_1 in synset_1_list:
			pos_1 = get_pos_synset(synset_1)
			for synset_2 in synset_2_list:
				pos_2 = get_pos_synset(synset_2)

				if(pos_1!=pos_2 and same_pos_comparison):
					continue
				if( (pos_1 in ['a','s','r'] or pos_2 in ['a','s','r']) and information_based_bool ):
					continue


				if(information_based_bool):
					score = similarity_function(synset_1,synset_2,brown_ic)
				else:
					score = similarity_function(synset_1,synset_2)	

				all_scores.append(score)

		
		all_scores = convert_none_to_zero(all_scores)
		if(pooling=='max'):
			score = np.max(all_scores)
		elif(pooling=='min'):
			score = np.min(all_scores)
		elif(pooling=='average'):
			score = np.mean(all_scores)	
		elif(pooling=='top'):
			score = all_scores[0]
		return score	

def get_data(data):
	if(data=='brown'):
		return brown
	elif(data=='gutenberg'):
		return gutenberg
	elif(data=='pros_cons'):
		return pros_cons
	elif(data=='movie_reviews'):
		return movie_reviews
	elif(data=='product_reviews_1'):
		return product_reviews_1
	elif(data=='product_reviews_2'):
		return product_reviews_2
	elif(data=='genesis'):
		return genesis
	elif(data=='reuters'):
		return reuters
	elif(data=='webtext'):
		return webtext	
	elif(data=='inaugural'):
		return inaugural


def get_documents(data):

	documents = []
 
	for fileid in data.fileids():
		document = ' '.join(data.words(fileid))
		documents.append(document)
	return documents

# credits: https://nlpforhackers.io/topic-modeling/
def clean_text(text):
	stopwords_list = stopwords.words('english')
	tokenized_text = word_tokenize(text.lower())
	cleaned_text = [t for t in tokenized_text if t not in stopwords_list and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
	return cleaned_text    

def tokenize_and_clean(documents):
	tokenized_data = []
	for text in documents:
		tokenized_data.append(clean_text(text)) 
	return tokenized_data





def get_similarity_object(s):
	if(s=='path'):
		return wn.path_similarity, False
	elif(s=='lch'):
		return wn.lch_similarity, False
	elif(s=='wup'):
		return wn.wup_similarity, False
	elif(s=='res'):
		return wn.res_similarity, True
	elif(s=='jcn'):
		return wn.jcn_similarity, True
	elif(s=='lin'):
		return wn.lin_similarity, True






brown_ic = wordnet_ic.ic('ic-brown.dat')

data_obj = get_data(args.dataset)
documents = get_documents(data_obj) 

tokenized_document = tokenize_and_clean(documents)
print("document length",len(tokenized_document))

dictionary = corpora.Dictionary(tokenized_document)

#print(dictionary.keys())
corpus = [dictionary.doc2bow(text) for text in tokenized_document]
bow_corpus = [dictionary.doc2bow(line) for line in tokenized_document]

print(bow_corpus)

term_doc_mat = corpus2csc(bow_corpus)
term_term_mat = np.dot(term_doc_mat, term_doc_mat.T)

word_to_idx = dictionary.token2id
idx_to_word = dictionary.id2token

with open('word_to_idx.pkl', 'wb') as f:
        pickle.dump(word_to_idx, f, pickle.HIGHEST_PROTOCOL)
with open('idx_to_word.pkl', 'wb') as f:
        pickle.dump(idx_to_word, f, pickle.HIGHEST_PROTOCOL)



vocab_len = len(dictionary.keys())
print("vocab_len",vocab_len)

word_word_matrix = sp.sparse.lil_matrix((vocab_len,vocab_len))
similarity_function, information_based_bool = get_similarity_object(args.similarity_function)



if(os.path.exists(args.similarity_function+".npy")):
	word_word_matrix = np.load(args.similarity_function+".npy")
else:	
	for i in prange(0,vocab_len,num_threads=4):
		print("i",i)
		w1 = idx_to_word[i]
		for j in range(i,vocab_len):
			w2 = idx_to_word[j]
			word_word_matrix[i][j] = get_score(w1,w2,args.pooling)
			word_word_matrix[j][i] = word_word_matrix[i][j]

	np.save(args.similarity_function+".npy",word_word_matrix)


