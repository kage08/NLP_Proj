from gensim.models import Word2Vec
from nltk.corpus import brown, reuters, gutenberg
from evaluation.funs import wordsim_eval
import numpy as np
from multiprocessing import Process, Queue
from embed import gensim_to_embed
import logging, gc, itertools

logging.basicConfig(filename='tests/eval/logs/wordsim.log', filemode='w', level=logging.INFO)

def train_skip(CBOW_skip=0, embed_size=100, window=5, min_count=5, epochs=5, workers=1):
    corpus_name = 'Brown'
    corpus = itertools.chain(reuters.sents(), brown.sents(), gutenberg.sents())
    corpus = list(corpus)
    model = Word2Vec(corpus,sg=CBOW_skip, size=embed_size, window=window, min_count=min_count, workers=workers)
    logging.warning("[1]"+",".join([corpus_name,str(CBOW_skip),str(embed_size),str(window)]))
    model.train(corpus, total_examples=len(corpus), epochs=epochs)
    model = gensim_to_embed(model)
    return wordsim_eval(model)

if __name__ == "__main__":
    print(train_skip())