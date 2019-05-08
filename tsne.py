from sklearn.manifold import TSNE
import pickle
import matplotlib.pyplot as plt
import numpy as np

rg = np.random.RandomState(10)

DICT = 'data/glove_3_jcn.pkl'
with open(DICT,'rb') as fl:
    embed = pickle.load(fl)
words = []
with open('data/cluster_words.txt','r') as fl:
    words = fl.readlines()
embed = embed[0].vocab
words = [w.strip() for w in words]

words = [w for w in words if w in embed.keys()]


ems = [embed[w] for w in words]

tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
new_values = tsne_model.fit_transform(ems)

plt.figure(figsize=(16, 16)) 

for i in range(len(ems)):
    plt.scatter(new_values[i][0], new_values[i][1])
    plt.annotate(words[i],xy=(new_values[i][0], new_values[i][1]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

plt.show()
