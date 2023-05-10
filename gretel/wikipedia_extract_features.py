import pandas as pd
import numpy as np
import networkx as nx
import fasttext
import os
import pickle

input_dir = '../workspace/wikispeedia/'
raw_input_dir = '../workspace/wikispeedia/raw_input_data/'

"""
Load data
"""
articles_df = pd.read_csv(os.path.join(raw_input_dir, 'articles.tsv'),
                          sep="\t",
                          comment='#',
                          header=None,
                          names=['article'])

links_df = pd.read_csv(os.path.join(raw_input_dir, 'links.tsv'),
                       sep='\t',
                       comment='#',
                       header=None,
                       names=['linkSource', 'linkTarget'])

categories_df = pd.read_csv(os.path.join(raw_input_dir, 'categories.tsv'),
                            sep='\t',
                            comment='#',
                            header=None,
                            names=['article', 'category'])

category_embeddings_df = pd.read_csv(os.path.join(raw_input_dir, "first_category_embeddings.tsv"),
                         sep="\t",
                         header=None,
                         names=['article', 'embeddings'])
ft = fasttext.load_model(os.path.join(input_dir, 'fastText/cc.en.300.bin'))
n_nodes = len(articles_df)
n_edges = len(links_df)

senders_id = []
receivers_id = []
senders_name = []
receivers_name = []
original_edge_features = {}
with open(os.path.join(input_dir, 'edges.txt'), 'r') as f:
    first_line = f.readline()
    for i, line in enumerate(f.readlines()):
        line = line.strip().split("\t")
        edge_id, out_id, in_id, tfidf, nof = int(line[0]), int(line[1]), int(line[2]), float(line[3]), float(line[4])
        senders_id.append(out_id)
        receivers_id.append(in_id)
        senders_name.append(articles_df.iloc[out_id][0])
        receivers_name.append(articles_df.iloc[in_id][0])
        original_edge_features[edge_id] = {}
        original_edge_features[edge_id]['out_id'] = out_id
        original_edge_features[edge_id]['in_id'] = in_id
        original_edge_features[edge_id]['tfidf'] = tfidf
        original_edge_features[edge_id]['nof'] = nof


original_node_features = {}
with open(os.path.join(input_dir, "nodes.txt"), 'r') as f:
    first_line = f.readline()
    for i, line in enumerate(f.readlines()):
        feat = []
        line = line.strip().split("\t")
        node_id, out_d, in_d = int(line[0]), float(line[1]), float(line[2])
        node_label = articles_df.iloc[node_id][0]
        original_node_features[node_id] = {}
        original_node_features[node_id]['label'] = node_label
        original_node_features[node_id]['out_degree'] = out_d
        original_node_features[node_id]['in_degree'] = in_d


with open(os.path.join(raw_input_dir, 'article_embeddings.txt'), 'r') as f:
    n_articles, d = 4604, 300
    article_embeddings = np.zeros([n_articles, d])
    for i, line in enumerate(f.readlines()):
        values = [float(x) for x in line.split()]
        article_embeddings[i] = values

#########################################################################################################
#########################################################################################################
#########################################################################################################
"""
Use one hot vector for the first category as edge feature.
The one hot vector from both edge's node is used.
The primal features also are used --> edges_original_onehot.txt --> 32 features
"""
num_features = 15 * 2 + 2  # 15 is the length of one hot vector for the first level category
with open(os.path.join(input_dir, "edges_original_onehot.txt"), "w") as f:
    f.write("{}\t{}\n".format(n_edges, num_features))
    for i in range(0, n_edges):
        feat = []
        out_id = original_edge_features[i]['out_id']
        in_id = original_edge_features[i]['in_id']
        out_node = articles_df.iloc[out_id][0]
        in_node = articles_df.iloc[in_id][0]
        out_embedding = category_embeddings_df.loc[category_embeddings_df['article'] == out_node]['embeddings'].apply(eval)[out_id] # transform series to list
        in_embedding = category_embeddings_df.loc[category_embeddings_df['article'] == in_node]['embeddings'].apply(eval)[in_id]

        original_edge_features[i]['out_embedding'] = out_embedding
        original_edge_features[i]['in_embedding'] = in_embedding

        feat.append(original_edge_features[i]['out_id'])
        feat.append(original_edge_features[i]['in_id'])
        feat.extend(out_embedding)
        feat.extend(in_embedding)
        feat.append(original_edge_features[i]['tfidf'])
        feat.append(original_edge_features[i]['nof'])
        line = str(i) + "\t" + "\t".join(map(str, [f for f in feat])) + "\n"
        f.write(line)

for i in range(0, n_nodes):
    node = articles_df.iloc[i][0]
    embedding = category_embeddings_df.loc[category_embeddings_df['article'] == node]['embeddings'].apply(eval)[i]
    original_node_features[i]['category_embedding'] = embedding

#########################################################################################################
#########################################################################################################
#########################################################################################################
"""
Use the similarity between the embedding of article's content 
and the embedding of the category label it belongs to as  edge feature.
The primal features also are used: 
edges_original_3categories.txt --> 8 features
edges_original_category1.txt   --> 4 features
edges_original_category2.txt   --> 4 features
edges_original_category3.txt   --> 4 features
"""
article_by_category = categories_df.drop_duplicates(subset='article').merge(articles_df, left_on='article', right_on='article', how='outer')
article_by_category.category[article_by_category.category == np.NaN] = ""
x = article_by_category['category'].str.split('.', expand=True)
article_by_category = x.groupby(article_by_category['article']).agg(lambda x: x)
article_by_category = article_by_category.drop(0, axis=1)
article_by_category.columns = ['category 1', 'category 2', 'category 3']
article_by_category = article_by_category.reset_index()
article_by_category = article_by_category.fillna("")

n_articles = article_by_category.shape[0]
d = 300
category_1 = np.zeros([n_articles, d])
category_2 = np.zeros([n_articles, d])
category_3 = np.zeros([n_articles, d])
for i in range(0, n_articles):
    category = article_by_category.iloc[i]['category 1']
    word_list = category.split('_')
    if len(word_list) == 1:
        category_1[i] = ft.get_word_vector(category)
    else:
        category = category.replace('_', ' ')
        category_1[i] = ft.get_sentence_vector(category)
    ################################################################
    category = article_by_category.iloc[i]['category 2']
    word_list = category.split('_')
    if len(word_list) == 1:
        category_2[i] = ft.get_word_vector(category)
    else:
        category = category.replace('_', ' ')
        category_2[i] = ft.get_sentence_vector(category)
    ################################################################
    category = article_by_category.iloc[i]['category 3']
    word_list = category.split('_')
    if len(word_list) == 1:
        category_3[i] = ft.get_word_vector(category)
    else:
        category = category.replace('_', ' ')
        category_3[i] = ft.get_sentence_vector(category)


distance_cat_1 = np.einsum('ij,ij->i', category_1, article_embeddings) / (np.linalg.norm(category_1, axis=1) * np.linalg.norm(article_embeddings, axis=1))
distance_cat_1[np.isnan(distance_cat_1)] = 0
distance_cat_1_norm = distance_cat_1 - min(distance_cat_1)
distance_cat_1_norm /= max(distance_cat_1_norm)


distance_cat_2 = np.einsum('ij,ij->i', category_2, article_embeddings) / (np.linalg.norm(category_2, axis=1) * np.linalg.norm(article_embeddings, axis=1))
distance_cat_2[np.isnan(distance_cat_2)] = 0
distance_cat_2_norm = distance_cat_2 - min(distance_cat_2)
distance_cat_2_norm /= max(distance_cat_2_norm)

distance_cat_3 = np.einsum('ij,ij->i', category_3, article_embeddings) / (np.linalg.norm(category_3, axis=1) * np.linalg.norm(article_embeddings, axis=1))
distance_cat_3[np.isnan(distance_cat_3)] = 0
distance_cat_3_norm = distance_cat_3 - min(distance_cat_3)
distance_cat_3_norm /= max(distance_cat_3_norm)


for i in range(0, n_nodes):
    original_node_features[i]['cat_1'] = distance_cat_1_norm[i]
    original_node_features[i]['cat_2'] = distance_cat_2_norm[i]
    original_node_features[i]['cat_3'] = distance_cat_3_norm[i]

num_features = 8
with open(os.path.join(input_dir, "edges_original_3categories.txt"), "w") as f:
    f.write("{}\t{}\n".format(n_edges, num_features))
    for i in range(0, n_edges):
        feat = []
        out_id = original_edge_features[i]['out_id']
        in_id = original_edge_features[i]['in_id']

        original_edge_features[i]['cat_1_out_id'] = distance_cat_1_norm[out_id]
        original_edge_features[i]['cat_2_out_id'] = distance_cat_2_norm[out_id]
        original_edge_features[i]['cat_3_out_id'] = distance_cat_3_norm[out_id]
        original_edge_features[i]['cat_1_in_id'] = distance_cat_1_norm[in_id]
        original_edge_features[i]['cat_2_in_id'] = distance_cat_2_norm[in_id]
        original_edge_features[i]['cat_3_in_id'] = distance_cat_3_norm[in_id]

        feat.append(original_edge_features[i]['out_id'])
        feat.append(original_edge_features[i]['in_id'])
        feat.append(original_edge_features[i]['tfidf'])
        feat.append(original_edge_features[i]['nof'])
        feat.append(original_edge_features[i]['cat_1_out_id'])
        feat.append(original_edge_features[i]['cat_2_out_id'])
        feat.append(original_edge_features[i]['cat_3_out_id'])
        feat.append(original_edge_features[i]['cat_1_in_id'])
        feat.append(original_edge_features[i]['cat_2_in_id'])
        feat.append(original_edge_features[i]['cat_3_in_id'])

        line = str(i) + "\t" + "\t".join(map(str, [f for f in feat])) + "\n"
        f.write(line)



num_features = 4
with open(os.path.join(input_dir, "edges_original_category1.txt"), "w") as f:
    f.write("{}\t{}\n".format(n_edges, num_features))
    for i in range(0, n_edges):
        feat = []
        feat.append(original_edge_features[i]['out_id'])
        feat.append(original_edge_features[i]['in_id'])
        feat.append(original_edge_features[i]['tfidf'])
        feat.append(original_edge_features[i]['nof'])
        feat.append(original_edge_features[i]['cat_1_out_id'])
        feat.append(original_edge_features[i]['cat_1_in_id'])
        line = str(i) + "\t" + "\t".join(map(str, [f for f in feat])) + "\n"
        f.write(line)


num_features = 4
with open(os.path.join(input_dir, "edges_original_category2.txt"), "w") as f:
    f.write("{}\t{}\n".format(n_edges, num_features))
    for i in range(0, n_edges):
        feat = []
        feat.append(original_edge_features[i]['out_id'])
        feat.append(original_edge_features[i]['in_id'])
        feat.append(original_edge_features[i]['tfidf'])
        feat.append(original_edge_features[i]['nof'])
        feat.append(original_edge_features[i]['cat_2_out_id'])
        feat.append(original_edge_features[i]['cat_2_in_id'])
        line = str(i) + "\t" + "\t".join(map(str, [f for f in feat])) + "\n"
        f.write(line)


num_features = 4
with open(os.path.join(input_dir, "edges_original_category3.txt"), "w") as f:
    f.write("{}\t{}\n".format(n_edges, num_features))
    for i in range(0, n_edges):
        feat = []
        feat.append(original_edge_features[i]['out_id'])
        feat.append(original_edge_features[i]['in_id'])
        feat.append(original_edge_features[i]['tfidf'])
        feat.append(original_edge_features[i]['nof'])
        feat.append(original_edge_features[i]['cat_3_out_id'])
        feat.append(original_edge_features[i]['cat_3_in_id'])
        line = str(i) + "\t" + "\t".join(map(str, [f for f in feat])) + "\n"
        f.write(line)


#########################################################################################################
#########################################################################################################
#########################################################################################################
"""
Use the similarity between two aritcles' embeddings.
The two articles represent the nodes of an edge.
edges_original_fasttextbody.txt  --> 3 features
"""
n_links, d = links_df.shape[0], 300
source_embeddings = np.zeros([n_links, d])  # create a table with the embeddings of source nodes
target_embeddings = np.zeros([n_links, d])  # create a table with the embeddings of target nodes
for i in range(0, links_df.shape[0]):
    source_embeddings[i] = article_embeddings[senders_id[i]]
    target_embeddings[i] = article_embeddings[receivers_id[i]]

# compute the cosine similarity between sender and target nodes
link_embeddings = np.einsum('ij,ij->i', source_embeddings, target_embeddings) / (np.linalg.norm(source_embeddings, axis=1) * np.linalg.norm(target_embeddings, axis=1))
link_embeddings[np.isnan(link_embeddings)] = 0
link_embeddings_norm = link_embeddings - min(link_embeddings)
link_embeddings_norm /= max(link_embeddings_norm)


num_features = 3
with open(os.path.join(input_dir, "edges_original_fasttextbody.txt"), "w") as f:
    f.write("{}\t{}\n".format(n_edges, num_features))
    for i in range(0, n_edges):
        feat = []
        original_edge_features[i]['fasttextbody_similarity'] = link_embeddings_norm[i]
        feat.append(original_edge_features[i]['out_id'])
        feat.append(original_edge_features[i]['in_id'])
        feat.append(original_edge_features[i]['tfidf'])
        feat.append(original_edge_features[i]['nof'])
        feat.append(original_edge_features[i]['fasttextbody_similarity'])
        line = str(i) + "\t" + "\t".join(map(str, [f for f in feat])) + "\n"
        f.write(line)

num_features = 2
with open(os.path.join(input_dir, "edges_nof_fasttextbody.txt"), "w") as f:
    f.write("{}\t{}\n".format(n_edges, num_features))
    for i in range(0, n_edges):
        feat = []
        feat.append(original_edge_features[i]['out_id'])
        feat.append(original_edge_features[i]['in_id'])
        feat.append(original_edge_features[i]['nof'])
        feat.append(original_edge_features[i]['fasttextbody_similarity'])
        line = str(i) + "\t" + "\t".join(map(str, [f for f in feat])) + "\n"
        f.write(line)

#########################################################################################################
#########################################################################################################
#########################################################################################################
"""
Use pagerank score as an edge feature.
The pagerank from both edge's node is used.
edges_original_page_rank.txt  --> 4 features
"""
G = nx.read_gml(os.path.join(input_dir, 'graph.gml'))
pr = nx.pagerank(G)
pr_values = np.fromiter(pr.values(), dtype=float)
pr_values_norm = pr_values - min(pr_values)
pr_values_norm /= max(pr_values_norm)
n_links = G.number_of_edges()

for i in range(0, n_nodes):
    original_node_features[i]['pagerank'] = pr_values_norm[i]

num_features = 4
with open(os.path.join(input_dir, "edges_original_page_rank.txt"), "w") as f:
    f.write("{}\t{}\n".format(n_edges, num_features))
    for i in range(0, n_edges):
        feat = []
        original_edge_features[i]['pagerank_out'] = pr_values_norm[senders_id[i]]
        original_edge_features[i]['pagerank_in'] = pr_values_norm[receivers_id[i]]
        feat.append(original_edge_features[i]['out_id'])
        feat.append(original_edge_features[i]['in_id'])
        feat.append(original_edge_features[i]['tfidf'])
        feat.append(original_edge_features[i]['nof'])
        feat.append(original_edge_features[i]['pagerank_out'])
        feat.append(original_edge_features[i]['pagerank_in'])
        line = str(i) + "\t" + "\t".join(map(str, [f for f in feat])) + "\n"
        f.write(line)

#########################################################################################################
#########################################################################################################
#########################################################################################################
"""
Use eigenvector_centrality score as an edge feature.
The eigenvector_centrality from both edge's node is used.
edges_original_centrality.txt  --> 4 features
"""
G = nx.read_gml(os.path.join(input_dir, 'graph.gml'))
centrality = nx.eigenvector_centrality(G)
centrality_values = np.fromiter(centrality.values(), dtype=float)
centrality_values_norm = centrality_values - min(centrality_values)
centrality_values_norm /= max(centrality_values_norm)
n_links = G.number_of_edges()

for i in range(0, n_nodes):
    original_node_features[i]['pagerank'] = centrality_values_norm[i]

num_features = 4
with open(os.path.join(input_dir, "edges_original_centrality.txt"), "w") as f:
    f.write("{}\t{}\n".format(n_edges, num_features))
    for i in range(0, n_edges):
        feat = []
        original_edge_features[i]['eigenvector_out'] = centrality_values_norm[senders_id[i]]
        original_edge_features[i]['eigenvector_in'] = centrality_values_norm[receivers_id[i]]
        feat.append(original_edge_features[i]['out_id'])
        feat.append(original_edge_features[i]['in_id'])
        feat.append(original_edge_features[i]['tfidf'])
        feat.append(original_edge_features[i]['nof'])
        feat.append(original_edge_features[i]['eigenvector_out'])
        feat.append(original_edge_features[i]['eigenvector_in'])
        line = str(i) + "\t" + "\t".join(map(str, [f for f in feat])) + "\n"
        f.write(line)


#########################################################################################################
#########################################################################################################
#########################################################################################################
"""
To test the sensitivity of Gretel to changes in node and edge features, 
a set of input files were created containing only the essential information:
node IDs and edge IDs along with the source and target node IDs for each edge.
These files don't include any extra features for either nodes or edges. 
Several experiments conducted to evaluate the impact of changes in node and edges 
features on the model's performance. The results showed that changes in node features
had no significant effect in the model's performance. 
Therefor all the proposed features are used as edge features in the Gretel model.
"""
with open(os.path.join(input_dir, "nodes_only.txt"), "w") as f:
    f.write("{}\t{}\n".format(n_nodes, 0))
    for i in range(0, n_nodes):
        line = str(i) + "\n"
        f.write(line)

with open(os.path.join(input_dir, "edges_only.txt"), "w") as f:
    f.write("{}\t{}\n".format(n_edges, 0))
    for i in range(0, n_edges):
        feat = []
        feat.append(original_edge_features[i]['out_id'])
        feat.append(original_edge_features[i]['in_id'])
        line = str(i) + "\t" + "\t".join(map(str, [f for f in feat])) + "\n"
        f.write(line)

#########################################################################################################
#########################################################################################################
#########################################################################################################
"""
Save the edge and node features in pickle files
"""
with open(os.path.join(input_dir, 'geolife_node_features.pickle'), 'wb') as f:
    pickle.dump(original_node_features, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(input_dir, 'original_edge_features.pickle'), 'wb') as f:
    pickle.dump(original_edge_features, f, protocol=pickle.HIGHEST_PROTOCOL)


# load nodes features
# with open(os.path.join(input_dir, 'geolife_node_features.pickle'), 'rb') as f:
#     nodes_features = pickle.load(f)


# load edge features
# with open(os.path.join(input_dir, 'original_edge_features.pickle'), 'rb') as f:
#     edge_features = pickle.load(f)



