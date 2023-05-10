import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import pickle

input_dir = '../workspace/gps/iWet'

"""
Load data
"""
iwet_edge_features = {}
with open(os.path.join(input_dir, "edges.txt")) as f:
    first_line = f.readline()
    for line in f:
        feat = []
        line = line.strip().split("\t")
        edge_id, out_id, in_id, feat1, feat2 = int(line[0]), int(line[1]), int(line[2]), float(line[3]), float(line[4])
        iwet_edge_features[edge_id] = {}
        iwet_edge_features[edge_id]['out_id'] = out_id
        iwet_edge_features[edge_id]['in_id'] = in_id
        iwet_edge_features[edge_id]['feat1'] = feat1
        iwet_edge_features[edge_id]['feat2'] = feat2

iwet_node_features = {}
with open(os.path.join(input_dir, "nodes.txt"), 'r') as f:
    first_line = f.readline()
    for i, line in enumerate(f.readlines()):
        feat = []
        line = line.strip().split("\t")
        node_id, feat1, feat2 = int(line[0]), float(line[1]), float(line[2])
        iwet_node_features[node_id] = {}
        iwet_node_features[node_id]['id'] = node_id
        iwet_node_features[node_id]['feat1'] = feat1
        iwet_node_features[node_id]['feat2'] = feat2

gps_edge_ids = {(iwet_edge_features[i]['out_id'], iwet_edge_features[i]['in_id']): i
                for i in range(0, len((iwet_edge_features)))}
n_nodes = 18317
n_edges = 43787


def compute_incidence_matrix(edges, n_nodes):
    n_edges = len(edges)
    incidence = np.zeros((n_nodes, n_edges))
    for edge_id, edge in enumerate(edges):
        incidence[edge[0], edge_id] = 1
        incidence[edge[1], edge_id] = 1
    return incidence


graph_incidence = compute_incidence_matrix(gps_edge_ids, n_nodes)
hypergraph_incidence = np.transpose(graph_incidence)


with open(os.path.join(input_dir, 'iWet_hyperedge_similarity.txt'), 'w') as f:
    f.write("{}\t{}\n".format(n_edges, 3))
    for i in range(0, n_edges):
        feat = []
        fr = iwet_edge_features[i]['out_id']
        to = iwet_edge_features[i]['in_id']
        hyperedge_fr = graph_incidence[fr, :].reshape(1, -1)
        hyperedge_to = graph_incidence[to, :].reshape(1, -1)
        sim = cosine_similarity(hyperedge_fr, hyperedge_to).reshape(-1)
        feat.append(fr)
        feat.append(to)
        feat.append(iwet_edge_features[i]['feat1'])
        feat.append(iwet_edge_features[i]['feat2'])
        feat.append(sim[0])
        iwet_edge_features[i]['hyper_similarity'] = sim[0]
        line = str(i) + "\t" + "\t".join(map(str, [f for f in feat])) + "\n"
        f.write(line)

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
    f = 0                                           # check for self-edges in the data
    t = 0
    for edge_id, edge in enumerate(edges):
        if incidence[edge[0], edge_id] == 0:        # check if the source node is already examined
            incidence[edge[0], edge_id] = -1        # there are no such cases
        if incidence[edge[1], edge_id] == 0:        # check if the target node is already examined as source node
            incidence[edge[1], edge_id] = 1         # if not, set value 1
        else:
            # print(edge_id, edge[0], edge[1])
            if incidence[edge[1], edge_id] == -1:
                f += 1          # count the times the current node was already as source node
            else:
                t += 1          # count the times the current node was already as target node
    print (f, t)                # (7, 0) 7 self-edges
    return incidence

graph_incidence_dir = compute_directed_incidence_matrix(gps_edge_ids, n_nodes)
hypergraph_incidence_dir = np.transpose(graph_incidence_dir)

n_hypernodes = n_edges
n_hyperedges = n_nodes

hypersenders = []
hyperreceivers = []
for i in range(0, n_hyperedges):
    hyperedge = hypergraph_incidence_dir[:, i]
    out_list = np.where(hyperedge == -1)[0]
    in_list = np.where(hyperedge == 1)[0]
    if len(in_list) > 0 and len(out_list) > 0:
        for in_element in in_list:
            for out_element in out_list:
                hypersenders.append(in_element)
                hyperreceivers.append(out_element)

in_degree = np.zeros(n_edges)
out_degree = np.zeros(n_edges)
nof_hypersenders = Counter(hypersenders)
nof_hyperreceivers = Counter(hyperreceivers)

for i in range(0, n_edges):
    in_degree[i] = nof_hyperreceivers[i]
    out_degree[i] = nof_hypersenders[i]

in_degree = in_degree / max(in_degree)
out_degree = out_degree / max(out_degree)

with open(os.path.join(input_dir, 'iWet_hyperedge_DNnode_in_out_degree.txt'), 'w') as f:
    f.write("{}\t{}\n".format(n_edges, 4))
    for i in range(0, n_edges):
        feat = []
        iwet_edge_features[i]['in_hyper_degree'] = in_degree[i]
        iwet_edge_features[i]['out_hyper_degree'] = out_degree[i]
        feat.append(iwet_edge_features[i]['out_id'])
        feat.append(iwet_edge_features[i]['in_id'])
        feat.append(iwet_edge_features[i]['feat1'])
        feat.append(iwet_edge_features[i]['feat2'])
        feat.append(iwet_edge_features[i]['in_hyper_degree'])
        feat.append(iwet_edge_features[i]['out_hyper_degree'])
        line = str(i) + "\t" + "\t".join(map(str, [f for f in feat])) + "\n"
        f.write(line)


with open(os.path.join(input_dir, 'iWet_hyperedge_similarity_DNnode_in_out_degree.txt'), 'w') as f:
    f.write("{}\t{}\n".format(n_edges, 5))
    for i in range(0, n_edges):
        feat = []
        feat.append(iwet_edge_features[i]['out_id'])
        feat.append(iwet_edge_features[i]['in_id'])
        feat.append(iwet_edge_features[i]['feat1'])
        feat.append(iwet_edge_features[i]['feat2'])
        feat.append(iwet_edge_features[i]['hyper_similarity'])
        feat.append(iwet_edge_features[i]['in_hyper_degree'])
        feat.append(iwet_edge_features[i]['out_hyper_degree'])
        line = str(i) + "\t" + "\t".join(map(str, [f for f in feat])) + "\n"
        f.write(line)

############################################################################################################
############################################################################################################
############################################################################################################
"""
Save the gps edge and node hyper-features in pickle files
"""
with open(os.path.join(input_dir, 'iwet_hyper_node_features.pickle'), 'wb') as f:
    pickle.dump(iwet_node_features, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(input_dir, 'iwet_hyper_edge_features.pickle'), 'wb') as f:
    pickle.dump(iwet_edge_features, f, protocol=pickle.HIGHEST_PROTOCOL)


# load nodes features
# with open(os.path.join(input_dir, 'iwet_hyper_node_features.pickle'), 'rb') as f:
#     iwet_gps_node_features = pickle.load(f)


# load edge features
# with open(os.path.join(input_dir, 'iwet_hyper_edge_features.pickle'), 'rb') as f:
#     iwet_gps_edge_features = pickle.load(f)


