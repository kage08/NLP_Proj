import pandas as pd
import numpy as np
import re
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
pat = re.compile(r'[^A-Za-z\s]')
df1 = pd.read_csv('data/twitter-sentiment-analysis2/train.csv', encoding = "ISO-8859-1")
df2 = pd.read_csv('data/twitter-sentiment-analysis2/test.csv', encoding = "ISO-8859-1")

df1 = df1.iloc[:,[2,1]]
df2 = df2.iloc[:,[1]]

d1,d2 = df1.to_numpy(), df2.to_numpy()

for i in range(len(d1)):
    d1[i,0] = pat.sub('',d1[i,0]).split()
    d1[i,0] = [lemmatizer.lemmatize(x) for x in d1[i,0]]
for i in range(len(d2)):
    d2[i,0] = pat.sub('',d2[i,0]).split()
    d2[i,0] = [lemmatizer.lemmatize(x) for x in d2[i,0]]

import pickle
with open('data/twitter-sentiment-analysis2/train.pkl','wb') as f:
    pickle.dump(d1,f)

with open('data/twitter-sentiment-analysis2/test.pkl','wb') as f:
    pickle.dump(d2,f)