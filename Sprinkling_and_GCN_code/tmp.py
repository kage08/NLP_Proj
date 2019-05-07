import pickle
import pandas as pd
a="aa"
with open('fullgraph.pkl', 'rb') as f:
    graph=pickle.load(f)

print("=====")
print(graph.nodes)

df = pd.read_csv("wordsim353/combined.csv")
word_1_list = list(df['Word 1'])
word_2_list = list(df['Word 2'])

graph_nodes = set(graph.nodes)
word_list = set(word_1_list).union(word_2_list)

print(len(graph_nodes.intersection(word_list)),len(word_list))