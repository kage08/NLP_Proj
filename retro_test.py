import pickle
from web.datasets.similarity import fetch_WS353
from retrofitting.retrofit import retrofit, retrofit_wnsim
from web.evaluate import evaluate_similarity, evaluate_on_all
from evaluation.funs import wordsim_eval

data = fetch_WS353()

emb = pickle.load(open('data/glove.pkl','rb'))
ds = pickle.load(open('data/glove_neigh.pkl','rb'))

for w in ds.keys():
    ds[w][0].extend(ds[w][1])
    ds[w][0].extend(ds[w][2])
    ds[w] = ds[w][0]


from embed import EmbedModel

emb_lch=retrofit_wnsim(emb, ds, 1,'lch') 

ans = EmbedModel(retrofit_wnsim(emb,ds, 10,'lch'))
#emb = EmbedModel(emb)

#print(wordsim_eval(emb))
print(wordsim_eval(ans))