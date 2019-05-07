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


def rank_list(array):
	array = -1.0*array
	temp = array.argsort()
	return np.arange(len(array))[temp.argsort()]


def compute_similarity(dictionary):
	df = pd.read_csv("wordsim353/combined.csv")
	word_1_list = list(df['Word 1'])
	word_2_list = list(df['Word 2'])
	word_pairs = zip(word_1_list,word_2_list)
	wordsim_353_scores = np.array(df['Human (mean)']).astype(float)

	refined_list_1 = []
	refined_list_2 = []
	y = []
	y_pred = []
	word_list = dictionary.keys()
	for i,(word_1,word_2) in enumerate(word_pairs):
		if(word_1 in word_list and word_2 in word_list):
			refined_list_1.append(word_1)
			refined_list_2.append(word_2)
			y.append(wordsim_353_scores[i] )
			emb_1 = dictionary[word_1]
			emb_2 = dictionary[word_2]
			y_pred.append(np.sum(emb_1*emb_2)/(np.linalg.norm(emb_1)*np.linalg.norm(emb_2)))

	df =  pd.DataFrame()
	df['word_1'] = refined_list_1
	df['word_2'] = refined_list_2
	df['y'] = y 
	df['y_pred'] = y_pred
	df['pred_rank'] = rank_list(np.array(y_pred))
	df['actual_rank'] = rank_list(np.array(y))
	df.to_csv("output_file.csv")		
	print(sp.stats.spearmanr(y, y_pred))








with open('embedding_300_no_sprinkling.pkl', 'rb') as f:
        dictionary = pickle.load(f)	
compute_similarity(dictionary)











