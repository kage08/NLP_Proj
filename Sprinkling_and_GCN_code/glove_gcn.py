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

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

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









with open('glove_300.pkl', 'rb') as f:
        full_glove_dict = pickle.load(f)

print(full_glove_dict["love"]) 

with open('fullgraph.pkl', 'rb') as f:
    graph=(pickle.load(f)).to_undirected()


nodes = list(graph.nodes())

common_words = list(set(nodes).intersection(set(full_glove_dict.keys())))
graph = graph.subgraph(common_words)


nodes = list(graph.nodes())

print("len",len(nodes))

word_2_idx_wwm={}
idx_2_word_wwm = {}

embedding_matrix = np.zeros((len(nodes),300))
for i,word in enumerate(nodes):
    word_2_idx_wwm[word] = i 
    idx_2_word_wwm[i] = word
    embedding_matrix[i] = full_glove_dict[word]



word_word_matrix = sp.sparse.lil_matrix((nx.to_scipy_sparse_matrix(graph)!=0).astype(float)+sp.sparse.identity(len(nodes)))   
word_word_matrix = sklearn.preprocessing.normalize(word_word_matrix,norm='l1', axis=1)

hops = 3

gcn_embedding = embedding_matrix
#print(gcn_embedding[0])
for i in range(hops):
    gcn_embedding = gcn_embedding + word_word_matrix.dot(gcn_embedding)
    print("shape",gcn_embedding.shape,len(graph.edges()))

#print(gcn_embedding[0]) 

output_dict = {}

for i,word in enumerate(nodes):
    output_dict[word] = gcn_embedding[i]

compute_similarity(word_2_idx_wwm,gcn_embedding)    

with open('embedding_300_glove_gcn_modified_hop_3.pkl', 'wb') as f:
        pickle.dump(output_dict, f, pickle.HIGHEST_PROTOCOL)    




