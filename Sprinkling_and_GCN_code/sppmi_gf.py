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
from sparsesvd import sparsesvd




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




word_2_idx_pmi_part = {}
idx_2_word_pmi_part = {}

with open("pmi.words.vocab") as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
avbl_words = [x.strip() for x in content]
avbl_vocab_length = len(avbl_words)

for i,word in enumerate(avbl_words):
	word_2_idx_pmi_part[word] = i
	idx_2_word_pmi_part[i] = word


with open('fullgraph.pkl', 'rb') as f:
    graph=(pickle.load(f)).to_undirected()

word_word_matrix = sp.sparse.lil_matrix(nx.to_scipy_sparse_matrix(graph))

tmpp_1 = word_word_matrix.dot(word_word_matrix)
tmpp_2 = word_word_matrix.dot(tmpp_1)
###tmpp_3 = word_word_matrix.dot(tmpp_2)

word_word_matrix = word_word_matrix ##+  tmpp_1 ###+ tmpp_2 ###+ tmpp_3
word_word_matrix = (word_word_matrix!=0).astype(float)



print("graph",graph)
nodes = list(graph.nodes())
print("nodes",nodes[:10])

old_mapping = {}
for i,word in enumerate(nodes):
	old_mapping[word] = i




common_words = list(set(nodes).intersection(avbl_words))
graph = graph.subgraph(common_words)
print("len",len(common_words),common_words[:5],list(graph.nodes())[:5])
word_2_idx = {}
idx_2_word = {}


for i,word in enumerate(common_words):
	word_2_idx[word] = i
	idx_2_word[i] = word

relevent_idx = [word_2_idx_pmi_part[i] for i in common_words]
relevent_idx_graph = [old_mapping[i] for i in common_words]


pmi_matrix = sp.sparse.lil_matrix(sp.io.loadmat("sparse_ppmi.mat")['ppmi'])
pmi_matrix = pmi_matrix[relevent_idx,:][:,relevent_idx]

pmi_matrix = sp.sparse.csc_matrix(pmi_matrix)


print("here!!")
graph_matrix = word_word_matrix[relevent_idx_graph,:][:,relevent_idx_graph]
graph_matrix = sp.sparse.csc_matrix(graph_matrix)
print("sum!!",graph_matrix.sum(),graph_matrix.shape,graph_matrix.diagonal().sum())
combined_matrix = sp.sparse.hstack((pmi_matrix, graph_matrix, graph_matrix))
###combined_matrix = graph_matrix
###combined_matrix = pmi_matrix

embeddingst, s, vt = sparsesvd(combined_matrix,k=300) 
embeddings = np.transpose(embeddingst)
#embeddings = embeddings[:,:args.latent_dim]
print("dot",np.dot(embeddings[0],embeddings[1]),embeddings.shape,s.shape)
sqrt_s = np.sqrt(s)

embeddings = np.matmul(embeddings,np.diag(sqrt_s))

embeddings_norm = np.linalg.norm(embeddings, ord=2, axis=1)
print("max",np.max(embeddings_norm),"min",np.min(embeddings_norm),np.where(embeddings_norm<1e-5)[0].shape)

filter_list =  np.where(embeddings_norm<1e-5)[0]

compute_similarity(word_2_idx,embeddings)
embeddings_dict = {}

for i in range(embeddings.shape[0]):
	if(i not in filter_list):
		embeddings_dict[idx_2_word[i]] = embeddings[i]
print("checks",len(embeddings_dict.keys()),len(common_words))		

with open('embedding_300_1_hop_sprinkling_twice.pkl', 'wb') as f:
        pickle.dump(embeddings_dict, f, pickle.HIGHEST_PROTOCOL)	






