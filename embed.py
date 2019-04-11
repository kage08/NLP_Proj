import numpy as np
from scipy.spatial.distance import cosine

class EmbedModel:

    def __init__(self, vocab = {}):
        self.vocab = vocab
    
    def __getitem__(self, str):
        return self.vocab[str]
    
    def size(self):
        return len(self.vocab)
    
    def get_sim(self, word1, word2):
        try:
            return 1-cosine(self.vocab[word1], self.vocab[word2])
        except KeyError:
            raise KeyError('Cant find either '+ word1+' or '+word2)
    
    def get_mostsimilar(self, embed):
        dist = np.infty
        ans = None

        for w in self.vocab.keys():
            d = 1-cosine(embed,self.vocab[w])
            if d<dist:
                d=dist
                ans = w
        
        return ans


def gensim_to_embed(model):
    vocab = {}
    for w in model.wv.vocab.keys():
        vocab[w] = model[w]
    embed = EmbedModel(vocab)
    return embed


        
    
