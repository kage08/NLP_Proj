import nltk
from nltk.corpus import wordnet as wn
from nltk.collocations import *
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
#import cPickle as pickle
import pickle
import networkx as nx
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


				score = wn.path_similarity(synset_1,synset_2)	

				all_scores.append(score)

		
		all_scores = convert_none_to_zero(all_scores)

		return np.max(all_scores)	

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





def compute_similarity(dictionary,embeddings):
	df = pd.read_csv("wordsim353/combined.csv")
	word_1_list = list(df['Word 1'])
	word_2_list = list(df['Word 2'])
	word_pairs = zip(word_1_list,word_2_list)

	wordsim_353_scores = np.array(df['Human (mean)'])
	word_1_list_converted = [dictionary[i] for i,j in zip(word_1_list,word_2_list) if i in dictionary.keys() and j in dictionary.keys()]
	word_2_list_converted = [dictionary[j] for i,j in zip(word_1_list,word_2_list) if i in dictionary.keys() and j in dictionary.keys()]

	embedding_1 = embeddings[word_1_list_converted]
	embedding_2 = embeddings[word_2_list_converted]
	
	wordsim_353_scores = [j for i,j in enumerate(wordsim_353_scores) if word_1_list[i] in dictionary.keys() and word_2_list[i] in dictionary.keys()]
	predicted_similarity = np.sum(embedding_1*embedding_2,axis=1)
	print(sp.stats.spearmanr(wordsim_353_scores, predicted_similarity))
	###print("max",np.sort(-1*predicted_similarity))













with open('fullgraph.pkl', 'rb') as f:
    graph=(pickle.load(f)).to_undirected()

print("graph",graph)
nodes = list(graph.nodes())

print("aaa1",list(graph.neighbors('love')) )

print("node_len",len(nodes),len(set(nodes)))



print("bbb",graph.degree['love'],nodes[15],nodes[17975],nodes[32301],nodes[32305],nodes[64779],nodes[65816],nodes[66119],nodes[66285],nodes[66286],nodes[66289])



word_2_idx_wwm={}
idx_2_word_wwm = {}
for i,word in enumerate(nodes):
	word_2_idx_wwm[word] = i 
	idx_2_word_wwm[i] = word


###graph = nx.relabel_nodes(graph, word_2_idx_wwm)


word_word_matrix = (nx.to_scipy_sparse_matrix(graph)!=0).astype(float)   
word_word_matrix = sklearn.preprocessing.normalize(word_word_matrix,norm='l1', axis=1)
###print(word_word_matrix.sum(axis=1)) 
print("aaa",(word_word_matrix[word_2_idx_wwm['love']]),word_word_matrix.sum(),word_2_idx_wwm['consummate'])
tmpp_1 = word_word_matrix.dot(word_word_matrix)
tmpp_2 = word_word_matrix.dot(tmpp_1)
tmpp_3 = word_word_matrix.dot(tmpp_2)
##tmpp_4 = word_word_matrix.dot(tmpp_3)
print("check",tmpp_1.sum(),word_word_matrix.sum())
word_word_matrix = word_word_matrix +  tmpp_1 + tmpp_2 + tmpp_3
word_word_matrix = (word_word_matrix!=0).astype(float)
###print("ccc",(word_word_matrix[word_2_idx_wwm['love']])[0,15],(word_word_matrix[word_2_idx_wwm['love']])[0,66286],(word_word_matrix[word_2_idx_wwm['love']])[0,66285])
#tmpp_2 = word_word_matrix.dot(word_word_matrix)

embeddings, s, vt = scipy.sparse.linalg.svds(word_word_matrix,k=args.latent_dim) 

sqrt_s = np.sqrt(s)
###print(s)
embeddings = np.matmul(embeddings,np.diag(sqrt_s))

###embeddings = sklearn.decomposition.TruncatedSVD(n_components=args.latent_dim).fit_transform(word_word_matrix)
print("dot",np.dot(embeddings[:,0],embeddings[:,1]),np.linalg.norm(embeddings[0]),np.dot(embeddings[1],embeddings[4]),embeddings.shape)
print("max",np.max(embeddings))

compute_similarity(word_2_idx_wwm,embeddings)
embeddings_dict = {}

for i in range(embeddings.shape[0]):
	embeddings_dict[idx_2_word_wwm[i]] = embeddings[i]

with open('three_hop.pkl', 'wb') as f:
        pickle.dump(embeddings_dict, f, pickle.HIGHEST_PROTOCOL)	






