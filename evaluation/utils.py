import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

'''
Get similarity data from csv file
'''
def get_data(path='data/wordsim353/combined.csv'):
    df = pd.read_csv(path, header=None)
    return df.to_numpy()

'''
Write file to csv
'''
def write_to_csv(arr,path):
    pd.DataFrame(arr).to_csv(path, header=False, index=False)

'''
Get Pearson and Spearman coefficients for scores
Both csvs should have same instances in same order (Add 0 to ignored ones)
'''
def get_corr(arr1, arr2, ignore_z = False, ifpath=False, extract=True):
    if ifpath:
        arr1, arr2 = get_data(arr1), get_data(arr2)
    if extract:
        arr1, arr2 = arr1[:,2].astype(np.float64), arr2[:,2].astype(np.float64)
    if ignore_z:
        idx = np.where(arr2!=0)
        arr1, arr2 = arr1[idx], arr2[idx]
    ans1 = pearsonr(arr1, arr2)
    ans2 = spearmanr(arr1, arr2)
    return ans1, ans2