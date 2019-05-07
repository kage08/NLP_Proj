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




word_2_idx = {}
idx_2_word = {}

with open("pmi.words.vocab") as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
vocabulary = [x.strip() for x in content]
vocab_length = len(vocabulary)

for i,word in enumerate(vocabulary):
	word_2_idx[word] = i
	idx_2_word[i] = word




pmi_matrix = sp.io.loadmat("sparse_ppmi.mat")['ppmi']





embeddingst, s, vt = sparsesvd(pmi_matrix,k=100) 
embeddings = np.transpose(embeddingst)
#embeddings = embeddings[:,:args.latent_dim]
print("dot",np.dot(embeddings[0],embeddings[1]),embeddings.shape,s.shape)
sqrt_s = np.sqrt(s)

embeddings = np.matmul(embeddings,np.diag(sqrt_s))

compute_similarity(word_2_idx,embeddings)
embeddings_dict = {}

for i in range(embeddings.shape[0]):
	embeddings_dict[idx_2_word[i]] = embeddings[i]

with open('test_embeds.pkl', 'wb') as f:
        pickle.dump(embeddings_dict, f, pickle.HIGHEST_PROTOCOL)	






