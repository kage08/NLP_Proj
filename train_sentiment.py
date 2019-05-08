import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

DICT_PATH = 'data/glove.pkl'
TRAIN_PATH = 'data/twitter-sentiment-analysis2/train.pkl'

EMBED_LEN = 300
SEED = 10
rg= np.random.RandomState(SEED)

with open(DICT_PATH,'rb') as f:
    embed = pickle.load(f)

with open(TRAIN_PATH,'rb') as f:
    train_data = pickle.load(f)

train_embeds, test_embeds = [],[]
train_idx, test_idx = [],[]
for i in range(len(train_data)):
    e = np.zeros(EMBED_LEN)
    ct = 0.
    for s in train_data[i,0]:
        if s in embed.keys():
            e+= embed[s]
            ct+=1
    if ct>0:
        e = e/ct
        train_idx.append(i)
        train_embeds.append(e)

train_labels = train_data[train_idx,1]



# Permute the dataset
perm = rg.permutation(len(train_embeds))
train_embeds = train_embeds[perm]
train_labels = train_labels[perm]

#Split Train/test
FRACTION = 0.8
t = int(len(train_embeds)*FRACTION)
train = np.array([train_embeds[:t], train_labels[:t]])
test = np.array([train_embeds[t:], train_labels[t:]])



