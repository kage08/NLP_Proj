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
	###print(len(dictionary.keys()),len(word_1_list),len(word_2_list),len(wordsim_353_scores))
	word_1_list_converted = [dictionary[i] for i,j in zip(word_1_list,word_2_list) if i in dictionary.keys() and j in dictionary.keys()]
	word_2_list_converted = [dictionary[j] for i,j in zip(word_1_list,word_2_list) if i in dictionary.keys() and j in dictionary.keys()]

	embedding_1 = embeddings[word_1_list_converted]
	embedding_2 = embeddings[word_2_list_converted]
	
	wordsim_353_scores = [j for i,j in enumerate(wordsim_353_scores) if word_1_list[i] in dictionary.keys() and word_2_list[i] in dictionary.keys()]
	predicted_similarity = np.sum(embedding_1*embedding_2,axis=1)
	print(sp.stats.spearmanr(wordsim_353_scores, predicted_similarity))










Corpus = get_data(args.dataset)
vocabulary = list(set(Corpus.words()))
vocab_len = len(vocabulary)

word_2_idx = {}
idx_2_word = {}


for i,word in enumerate(vocabulary):
	word_2_idx[word] = i 
	idx_2_word[i] = word

#print(len(Corpus.words()),len(set(Corpus.words())))

bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(Corpus.words(),window_size=2)

finder.apply_freq_filter(3)

#print(finder.score_ngrams(bigram_measures.pmi)[:100])

pmi_matrix = sp.sparse.lil_matrix((vocab_len,vocab_len))


with open('fullgraph.pkl', 'rb') as f:
    graph=(pickle.load(f)).to_undirected()

nodes = list(graph.nodes())
#print(nodes)
###print("edges",list(graph.edges()))

#print("abb",(nx.to_scipy_sparse_matrix(graph)[word_2_idx['love']]!=0).sum()) 
print("aaa1",list(graph.neighbors('love')) )

print("node_len",len(nodes),len(set(nodes)))

relevent_nodes = set(nodes).intersection(set(vocabulary))

graph = graph.subgraph(list(relevent_nodes))

print("bbb",graph.degree['love'])


reduced_vocab = list(graph.nodes())
reduced_vocab_len = len(reduced_vocab)
###print(reduced_vocab)


word_2_idx_wwm={}
idx_2_word_wwm = {}
for i,word in enumerate(reduced_vocab):
	word_2_idx_wwm[word] = i 
	idx_2_word_wwm[i] = word
print('id' in word_2_idx_wwm,'id' in reduced_vocab)
graph = nx.relabel_nodes(graph, word_2_idx_wwm)



pmi_matrix = sp.sparse.lil_matrix((reduced_vocab_len,reduced_vocab_len))
word_word_matrix = nx.to_scipy_sparse_matrix(graph)    
print("aaa",(word_word_matrix[word_2_idx_wwm['love']]),word_word_matrix.sum())
 







'''for i in prange(0,vocab_len,num_threads=8):
	print("i",i)
	w1 = idx_2_word[i]
	for j in range(i,vocab_len):
		#print("j",j)
		w2 = idx_2_word[j]
		word_word_matrix[i,j] = get_score(w1,w2,args.pooling)
		word_word_matrix[j,i] = word_word_matrix[i,j]
'''


for word_pair,pmi in finder.score_ngrams(bigram_measures.pmi):
	if(word_pair[0] in reduced_vocab and word_pair[1] in reduced_vocab):
		#####print("Here")
		######pmi = max(pmi - np.log(5),0)
		idx0 = word_2_idx_wwm[word_pair[0]]
		idx1 = word_2_idx_wwm[word_pair[1]]
		pmi_matrix[idx0,idx1] = pmi
		pmi_matrix[idx1,idx0] = pmi

pmi_matrix = sp.sparse.hstack((pmi_matrix, word_word_matrix))
####pmi_matrix = word_word_matrix

#pmi_matrix = pmi_matrix - math.log(5)


embeddings, s, vt = scipy.sparse.linalg.svds(pmi_matrix,k=args.latent_dim) 
#embeddings = embeddings[:,:args.latent_dim]
print("dot",np.dot(embeddings[0],embeddings[1]),embeddings.shape,s.shape)
sqrt_s = np.sqrt(s)

embeddings = np.matmul(embeddings,np.diag(sqrt_s))

compute_similarity(word_2_idx_wwm,embeddings)
embeddings_dict = {}

for i in range(embeddings.shape[0]):
	embeddings_dict[idx_2_word_wwm[i]] = embeddings[i]

with open('test_embeds.pkl', 'wb') as f:
        pickle.dump(embeddings_dict, f, pickle.HIGHEST_PROTOCOL)	






