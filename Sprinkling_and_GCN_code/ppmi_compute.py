import numpy as np
import scipy as sp
import scipy.io
import scipy.sparse

k = 5 

loader = np.load("pmi.npz")
b = scipy.sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
b.data = np.log(b.data)
b.data -= np.log(k)
b.data[b.data < 0] = 0
b.eliminate_zeros()

dict = {}

dict['ppmi']=b
sp.io.savemat("sparse_ppmi.mat",dict)
