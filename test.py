import pickle

with open('data/glove.pkl','rb') as fl:
    glove_dict = pickle.load(fl)

with open('data/fullgraph.pk','rb') as fl:
    graph = pickle.load(fl)

import networkx as nx
def get_n_hop(g,w,n):
    ans = [[] for i in range(n)]
    a = nx.single_source_shortest_path_length(g,w, cutoff=n)
    for w in a.keys():
        if a[w]<1:
            continue
        ans[a[w]-1].append(w)
    return ans

ds = {}
s = set(glove_dict.keys()).intersection(set(graph.nodes))
print('Len:', len(s))
input('Ichso')
ct=0
for w in set(glove_dict.keys()).intersection(set(graph.nodes)):
    ds[w] = get_n_hop(graph, w, 4)
    print(ct)
    ct = ct+1
    

with open('data/glove_neigh.pkl','wb') as fl:
    pickle.dump(ds, fl)