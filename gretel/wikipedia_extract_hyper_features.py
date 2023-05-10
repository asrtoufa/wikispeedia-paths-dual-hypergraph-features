import os
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

input_dir = '../workspace/wikispeedia/'
raw_input_dir = '../workspace/wikispeedia/raw_input_data'

"""
Load data
"""
articles_df = pd.read_csv(os.path.join(raw_input_dir, 'articles.tsv'),
                          sep='\t',
                          comment='#',
                          header=None,
                          names=['article'])

links_df = pd.read_csv(os.path.join(raw_input_dir, 'links.tsv'),
                       sep='\t',
                       comment='#',
                       header=None,
                       names=['linkSource', 'linkTarget'])

article_to_id = {a: i for i, a in enumerate(articles_df.article.values)}
id_to_article = {i: a for i, a in enumerate(articles_df.article.values)}
edge_ids = {(article_to_id[s], article_to_id[r]): i
            for i, (s, r) in enumerate(zip(links_df.linkSource, links_df.linkTarget))}

n_nodes = len(article_to_id)
n_edges = len(edge_ids)

n_hypernodes = n_edges
n_hyperedges = n_nodes


with open(os.path.join(input_dir, 'geolife_node_features.pickle'), 'rb') as f:
    original_node_features = pickle.load(f)

with open(os.path.join(input_dir, 'original_edge_features.pickle'), 'rb') as f:
    original_edge_features = pickle.load(f)


#########################################################################################################
#########################################################################################################
#########################################################################################################
"""
Compute Incidence matrix of the graph
and extract similarity of incidence vectors
"""
def compute_incidence_matrix(edges, n_nodes):
    n_edges = len(edges)
    incidence = np.zeros((n_nodes, n_edges))
    for edge_id, edge in enumerate(edges):
        incidence[edge[0], edge_id] = 1
        incidence[edge[1], edge_id] = 1
    return incidence


graph_incidence = compute_incidence_matrix(edge_ids, n_nodes)  # graph_incidence.shape = (4604, 119882)
hypergraph_incidence = np.transpose(graph_incidence)      # hypergraph_incidence.shape = (119882, 4604)


with open(os.path.join(input_dir, 'edges_original_hyperedge_similarity.txt'), 'w') as f:
    f.write("{}\t{}\n".format(n_edges, 3))
    for i in range(0, n_edges):
        feat = []
        fr = original_edge_features[i]['out_id']
        to = original_edge_features[i]['in_id']
        hyperedge_fr = graph_incidence[fr, :].reshape(1, -1)
        hyperedge_to = graph_incidence[to, :].reshape(1, -1)
        sim = cosine_similarity(hyperedge_fr, hyperedge_to).reshape(-1)
        feat.append(fr)
        feat.append(to)
        feat.append(original_edge_features[i]['tfidf'])
        feat.append(original_edge_features[i]['nof'])
        feat.append(sim[0])
        original_edge_features[i]['hyper_similarity'] = sim[0]
        line = str(i) + "\t" + "\t".join(map(str, [f for f in feat])) + "\n"
        f.write(line)


# check if similarity need normalization
similarity = []
for i in range(0, n_edges):
    similarity.append(original_edge_features[i]['hyper_similarity'])

# no extra normalization is needed
max(similarity)  # 1.0000000000000004
min(similarity)  # 0.0006895240031534343

############################################################################################################
############################################################################################################
############################################################################################################
"""
Create the incidence matrix of the directed version of the graph
Compute the in/out degree of each node in dual hypergraph
"""
def compute_directed_incidence_matrix(edges, n_nodes):
    n_edges = len(edges)
    incidence = np.zeros((n_nodes, n_edges))
    f = 0                                       # check for self-edges in the data
    t = 0
    for edge_id, edge in enumerate(edges):
        if incidence[edge[0], edge_id] == 0:    # check if the source node is already examined
            incidence[edge[0], edge_id] = -1    # there are no such cases
        if incidence[edge[1], edge_id] == 0:    # check if the target node is already examined as source node
            incidence[edge[1], edge_id] = 1     # if not, set value 1
        else:
            # print(edge_id, edge[0], edge[1])
            if incidence[edge[1], edge_id] == -1:
                f += 1      # count the times the current node was already as source node
            else:
                t += 1      # count the times the current node was already as target node
    print(f, t)             # (110, 0) 110 self-edges
    return incidence


graph_incidence = compute_directed_incidence_matrix(edge_ids, n_nodes)  # graph_incidence.shape     = (4604, 119882)
hypergraph_incidence = np.transpose(graph_incidence)                    #hypergraph_incidence.shape = (119882, 4604)

hypersenders = []
hyperreceivers = []
for i in range(0, n_hyperedges):
    hyperedge = hypergraph_incidence[:, i]      # hyperedge.shape = (119882,)
    out_list = np.where(hyperedge == -1)[0]     # list with sender nodes
    in_list = np.where(hyperedge == 1)[0]       # list with receiver nodes
    if len(in_list) > 0 and len(out_list) > 0:
        for in_element in in_list:              # compute the combinations among the connected nodes in dual hypergraph
            for out_element in out_list:
                hypersenders.append(in_element)
                hyperreceivers.append(out_element)
# len(hypersenders)     = 6909695
# len(hyperreceivers)   = 6909695

in_degree = np.zeros(n_edges)
out_degree = np.zeros(n_edges)
nof_hypersenders = Counter(hypersenders)      # count how many times each node of the dual hypergraph is a sender node
nof_hyperreceivers = Counter(hyperreceivers)  # count how many times each node of the dual hypergraph is a receiver node
for i in range(0, n_edges):
    in_degree[i] = nof_hyperreceivers[i]
    out_degree[i] = nof_hypersenders[i]

# normalize in/out degree
in_degree = in_degree / max(in_degree)
out_degree = out_degree / max(out_degree)

with open(os.path.join(input_dir, 'edges_original_hyperedge_in_out_degree.txt'), 'w') as f:
    f.write("{}\t{}\n".format(n_edges, 4))
    for i in range(0, n_edges):
        feat = []
        original_edge_features[i]['in_hyper_degree'] = in_degree[i]
        original_edge_features[i]['out_hyper_degree'] = out_degree[i]
        feat.append(original_edge_features[i]['out_id'])
        feat.append(original_edge_features[i]['in_id'])
        feat.append(original_edge_features[i]['tfidf'])
        feat.append(original_edge_features[i]['nof'])
        feat.append(in_degree[i])
        feat.append(out_degree[i])
        line = str(i) + "\t" + "\t".join(map(str, [f for f in feat])) + "\n"
        f.write(line)


############################################################################################################
with open(os.path.join(input_dir, 'edges_original_hyperedge_similarity_in_out_degree.txt'), 'w') as f:
    f.write("{}\t{}\n".format(n_edges, 5))
    for i in range(0, n_edges):
        feat = []
        feat.append(original_edge_features[i]['out_id'])
        feat.append(original_edge_features[i]['in_id'])
        feat.append(original_edge_features[i]['tfidf'])
        feat.append(original_edge_features[i]['nof'])
        feat.append(original_edge_features[i]['hyper_similarity'])
        feat.append(original_edge_features[i]['in_hyper_degree'])
        feat.append(original_edge_features[i]['out_hyper_degree'])
        line = str(i) + "\t" + "\t".join(map(str, [f for f in feat])) + "\n"
        f.write(line)


############################################################################################################
############################################################################################################
############################################################################################################
"""
Save the edge and node hyper-features in pickle files
"""
with open(os.path.join(input_dir, 'original_hyper_node_features.pickle'), 'wb') as f:
    pickle.dump(original_node_features, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(input_dir, 'original_hyper_edge_features.pickle'), 'wb') as f:
    pickle.dump(original_edge_features, f, protocol=pickle.HIGHEST_PROTOCOL)


# load nodes features
# with open(os.path.join(input_dir, 'original_hyper_node_features.pickle'), 'rb') as f:
#     nodes_features = pickle.load(f)


# load edge features
# with open(os.path.join(input_dir, 'original_hyper_edge_features.pickle'), 'rb') as f:
#     edge_features = pickle.load(f)
