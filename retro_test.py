import pickle
from web.datasets.similarity import fetch_WS353
from retrofitting.retrofit import retrofit, retrofit_wnsim
from web.evaluate import evaluate_similarity, evaluate_on_all
from evaluation.funs import wordsim_eval
from embed import EmbedModel
from copy import deepcopy
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

methods = ['simple','path', 'lch','jcn','lin', 'wup']
hops = [1,2,3]
res_file = open('data/results_wn.csv','a')


emb = pickle.load(open('data/glove.pkl','rb'))
ds_ = pickle.load(open('data/glove_wn.pkl','rb'))

for m in methods:
    for h in hops:
        name = '_'.join(['glove',str(h),m])
        print('Doing ',name)
        ds = deepcopy(ds_)
        for w in ds.keys():
            for i in range(1,h):
                ds[w][0].extend(ds[w][h])
            ds[w] = ds[w][0]
        
        if m == 'simple':
            new_embed = retrofit(emb, ds, 20)
            new_embed = (EmbedModel(new_embed), None)
        else:
            new_embed = retrofit_wnsim(emb, ds, 20, m)
            new_embed = (EmbedModel(new_embed[0]), new_embed[1])
        print('Calculating Results ...')
        results = evaluate_on_all(new_embed[0].vocab)
        print(results)
        line = [name]
        line.extend([str(x) for x in np.array(results)[0]])
        res_file.write(','.join(line))
        res_file.write('\n')
        res_file.flush()
        with open('data/'+name+'.pkl','wb') as fl:
            pickle.dump(new_embed, fl)
        del ds
        del new_embed
        print('\n#################################################')
