from evaluation.utils import get_data, get_corr
from scipy.spatial.distance import cosine
import numpy as np

def wordsim_eval(embed, wordsim_path='data/wordsim353/combined.csv', ignoe_z=True):
    ws_data = get_data(wordsim_path)
    simscores = []
    for i in range(ws_data.shape[0]):
        try:
            simscores.append([float(ws_data[i,2]), embed.get_sim(ws_data[i,0], ws_data[i,1])])
        except KeyError:
            simscores.append([float(ws_data[i,2]), 0])
    
    simscores = np.array(simscores, dtype=np.float)

    return get_corr(simscores[:,0], simscores[:,1], extract=False, ignore_z=ignoe_z)


