import pandas as pd
import networkx as nx
import os
import numpy as np
from graph import Graph
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import matplotlib.pyplot as plt
import re


raw_input_dir = '../workspace/wikispeedia/raw_input_data/'
output_dir = '../workspace/wikispeedia/'


"""
Load dataframes
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

paths_df = pd.read_csv(os.path.join(raw_input_dir, 'paths_finished.tsv'),
                       sep='\t',
                       comment='#',
                       header=None,
                       names=['hashedIpAddress', 'timestamp', 'durationInSec', 'path', 'rating'])

article_to_id = {a: i for i, a in enumerate(articles_df.article.values)}
id_to_article = {i: a for i, a in enumerate(articles_df.article.values)}
edge_ids = {(article_to_id[s], article_to_id[r]): i
            for i, (s, r) in enumerate(zip(links_df.linkSource, links_df.linkTarget))}


ill_paths = [2395, 4375, 8046, 9794, 17848, 48079]
paths_df = paths_df.drop(ill_paths)
paths_df = paths_df.reset_index()


senders = torch.tensor(links_df.linkSource.map(article_to_id.__getitem__))
receivers = torch.tensor(links_df.linkTarget.map(article_to_id.__getitem__))

graph = Graph(senders=senders, receivers=receivers, nodes=None, edges=None,
              n_node=len(articles_df), n_edge=len(links_df))

"""
Save Trajectories
"""
trajectory_articles = []
for path in paths_df.path.values:
    articles = path.split(';')
    direct_path = []
    # remove backtracking
    for a in articles:
        if a == '<':
            direct_path.pop()
        else:
            direct_path.append(a)

    trajectory_articles.append(direct_path)
# trajectory articles = (list:51312)
# trajectory articles[0] ={list:9} ['14th_century', '15th_century', '16th_century', 'Pacific_Ocean', 'Atlantic_Ocean', 'Accra', 'Africa', 'Atlantic_slave_trade', 'African_slave_trade']
# trajectory_articles[1] ={list:5} ['14th_century', 'Europe', 'Africa', 'Atlantic_slave_trade', 'African_slave_trade']
# trajectory_articles[2] ={list:8} ['14th_century', 'Niger', 'Nigeria', 'British_Empire', 'Slavery', 'Africa', 'Atlantic_slave_trade', 'African_slave_trade']


# save length
# for every path it saves its length, namely how many links there are in the path
# first value is the id and the second one the length of the path
# there are 51312 records in total
# 0	9
# 1	5
# 2	8
# 3	4
# 4	7
# 5	6
with open(os.path.join(output_dir, 'lengths.txt'), 'w') as f:
    for i, articles in enumerate(trajectory_articles):
        f.write("{}\t{}\n".format(i, len(articles)))


# save observations
# for each path, it saves the article's id
# In the first line there are the total number of observations
# 305210	1
# 10	1.0
# 12	1.0
# 15	1.0
# 3134	1.0
# 377	1.0
# 105	1.0
with open(os.path.join(output_dir, 'observations.txt'), 'w') as f:
    print(list(map(len, trajectory_articles)))   # [9, 5, 8, 4, 7, 6, 4, 6, 4, 7, 11, 10, 5....]
    num_observations = sum(map(len, trajectory_articles))   # 305210
    f.write("{}\t{}\n".format(num_observations, 1))
    for articles in trajectory_articles:
        for article in articles:
            f.write("{}\t{}\n".format(article_to_id[article], 1.))


# save paths
# computed only on the train_dataset
# it saves the id of traversed edges
traversed_edge_count = np.zeros(graph.n_edge)
test_dataset_start = int(len(trajectory_articles) * 0.8)
print("edge counts only until index {}/{}".format(test_dataset_start, len(trajectory_articles)))
with open(os.path.join(output_dir, 'paths.txt'), 'w') as f:
    num_paths = sum(map(len, trajectory_articles)) - len(trajectory_articles)  # 305210 - 51312 = 253898
    f.write("{}\t{}\n".format(num_paths, 1))
    for i, articles in enumerate(trajectory_articles):  # for each path
        for fr, to in zip(articles, articles[1:]):  # for each transition/link in a path
            # example:
            # trajectory_articles[0] = ['14th_century','15th_century','16th_century','Pacific_Ocean','Atlantic_Ocean','Accra','Africa','Atlantic_slave_trade','African_slave_trade']
            # trajectory_articles[0][1:] = ['15th_century','16th_century','Pacific_Ocean','Atlantic_Ocean','Accra','Africa','Atlantic_slave_trade','African_slave_trade']
            # the iteration is:
            # from =  14th_century to =  15th_century
            # from =  15th_century to =  16th_century
            # from =  16th_century to =  Pacific_Ocean
            # etc
            try:
                # every edge of initial graph from links.tsv is saved in edge_ids
                # edge_ids = dict with key a tuple (linkSource, linksTarget) and value the edge's id (0,1,2....119881)
                # article_to_id[fr] = linkSource id
                # article_to_id[to] = linksTarget id
                # edge variable is the id of graph edge
                edge = edge_ids[(article_to_id[fr], article_to_id[to])]
                if i < test_dataset_start:
                    # it counts how many times a links appears into paths
                    traversed_edge_count[edge] += 1
            except KeyError as e:
                print('no link between {} -> {} for traj {}'.format(fr, to, i))
                edge = 0
            f.write("{}\n".format(edge))



"""
Extract categories
"""
categories_df = pd.read_csv(os.path.join(raw_input_dir, 'categories.tsv'), sep='\t', comment='#', header=None, names=['article', 'category'])
article_by_category = categories_df.drop_duplicates(subset='article').merge(articles_df, left_on='article', right_on='article', how='outer')
article_by_category.category[article_by_category.category == np.NaN] = ""
article_by_category.head(20)

def category_extractor(cat):
    if cat is np.NaN:
        return {}
    return {"category{}".format(i): c for i, c in enumerate(cat.split('.'))}


"""
Feature extraction - TF-IDF
"""
tf_idf = TfidfVectorizer('filename')
files = ["{}/plaintext_articles/{}.txt".format(output_dir, article) for article in articles_df.article.values]
tfidf_vectors = tf_idf.fit_transform(files)
distances = cosine_similarity(tfidf_vectors)

plt.imshow(distances)
plt.colorbar()


def closest_articles(article, distances, k=5):
    id_ = article_to_id[article]
    distance_others = distances[id_]
    sorted_indices = np.argsort(distance_others)[::-1]
    sorted_indices = sorted_indices[sorted_indices != id_]
    top_ids = sorted_indices[:k]
    return list(zip(map(id_to_article.__getitem__, top_ids), distances[id_][top_ids]))


closest_articles('Moon', distances)

"""
Fast text embeddings
"""
with open(os.path.join(raw_input_dir, 'article_embeddings.txt'), 'r') as f:
    n_articles, d = 4604, 300
    emb = np.zeros([n_articles, d])
    for i, line in enumerate(f.readlines()):
        values = [float(x) for x in line.split()]
        emb[i] = values

emb_distances = cosine_similarity(emb)

plt.imshow(emb_distances)
plt.colorbar()
closest_articles("Moon", emb_distances)

emb_distances_norm = emb_distances - emb_distances.min()
emb_distances_norm /= emb_distances_norm.max()
plt.imshow(emb_distances_norm)
plt.colorbar()

"""
Add features to graph
"""
# NODE FEATURES
embeddings = torch.tensor(emb).float()
embeddings_norm = embeddings.norm(dim=1)
out_d = graph.out_degree_counts.float()
in_d = graph.in_degree_counts.float()

graph.nodes = torch.cat([
    out_d.unsqueeze(1) / out_d.max(),
    in_d.unsqueeze(1) / in_d.max(),
#     embeddings,
#     embeddings_norm.unsqueeze(1),
#     1. / embeddings_norm.unsqueeze(1)
], dim=1)


# EDGE FEATURES
fasttext_distances = torch.tensor(emb_distances_norm).float()
tfidf_distances = torch.tensor(distances).float()
traversed_edge_feature = torch.tensor(1. * traversed_edge_count / traversed_edge_count.max()).float()

graph.edges = torch.stack([
#     fasttext_distances[graph.senders, graph.receivers],
    tfidf_distances[graph.senders, graph.receivers],
    traversed_edge_feature
], dim=1)


graph.write_to_directory(output_dir)

"""
Given as target 
"""
targets = [t[-1] for t in trajectory_articles]
given_as_target = torch.zeros(graph.n_node, dtype=torch.uint8)
for target in targets:
    given_as_target[article_to_id[target]] = 1

torch.save(given_as_target, os.path.join(output_dir, "given_as_target.pt"))


"""
Node target preprocessing
"""
pairwise_features = torch.cat([
    fasttext_distances.unsqueeze(-1),
    tfidf_distances.unsqueeze(-1)
], dim=-1)

torch.save(pairwise_features, os.path.join(output_dir, "pairwise_node_features.pt"))

"""
Siblings nodes
"""
all_categories = sorted(list(set(categories_df.category)))
category_to_id = {c: i for i, c in enumerate(all_categories)}

# -- n_article, n_cat
article_category_one_hot = torch.zeros([len(articles_df), len(all_categories)], dtype=torch.uint8)
for i, row in categories_df.iterrows():
    article_category_one_hot[article_to_id[row.article], category_to_id[row.category]] = 1

siblings = torch.einsum("ac,bc->ab", [article_category_one_hot.float(), article_category_one_hot.float()])
siblings += torch.eye(siblings.shape[0])
siblings = siblings > 0

torch.save(siblings, os.path.join(output_dir, "siblings.pt"))


"""
Export to .gml
"""
graph = nx.DiGraph()
graph.add_nodes_from([(row.article, {'id': i, 'article': row.article, **category_extractor(row.category)}) for i, (_, row) in enumerate(article_by_category.iterrows())])
article_to_id_f = article_to_id.__getitem__
graph.add_edges_from(zip(links_df.linkSource, links_df.linkTarget, map(lambda x: {'Weigth': int(x)}, traversed_edge_count)))

# Sample some path
num_samples = 20
sampled_paths = np.random.choice(trajectory_articles, num_samples)

# for i in range(len(sampled_paths)):
#     nx.set_node_attributes(graph, 0, "path{}".format(i))
for i, path in enumerate(sampled_paths):
    path_name = 'pathx{}x{}x{}'.format(len(path), path[0], path[-1])
    path_name = re.sub(r'\W+', '', path_name).replace("_", "")
    for step, node in enumerate(path):
        graph.nodes[node][path_name] = (step + 1) * 10

nx.write_gml(graph, os.path.join(output_dir, 'graph.gml'))


"""
Create dataframe with article's name and one hot vector for the first level category
example:
10th_century	[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
11th_century	[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
12th_century	[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
"""
def get_multi_hot(s, col):
    if col == 1:
        return [1 if c in s else 0 for c in cats_1]
    else:
        print("Wrong argument provided")


categories_df = pd.read_csv(os.path.join(raw_input_dir, 'categories.tsv'),
                            sep='\t',
                            comment='#',
                            header=None,
                            names=['article', 'category'])
# categories_df.shape = (5204, 2)  # some articles belong to more than one categories
categories_splitted_df = categories_df['category'].str.split('.', expand=True)      # dataframe only with the splitted categories
articles_categories_df = categories_splitted_df.groupby(categories_df['article']).agg(lambda x: set(x))    # new_df.shape = (4598, 4)

cats_1 = categories_splitted_df[1].unique()
cats_1 = [a for a in cats_1 if a is not None]


articles_categories_df[1] = articles_categories_df[1].apply(get_multi_hot, args=[1])
articles_categories_df[0] = articles_categories_df[1]
del articles_categories_df[1]
del articles_categories_df[2]
del articles_categories_df[3]
articles_categories_df = articles_categories_df.rename(columns={0: 'embedding'})

first_category_embeddings = articles_df.merge(articles_categories_df, on=['article'], how='left', indicator=True)
# check for articles without categories - set one hot vector with zeros to all categories
first_category_embeddings.loc[first_category_embeddings['_merge'] == 'left_only', 'embedding'] = 0
first_category_embeddings['embedding'] = first_category_embeddings['embedding'].apply(lambda x: x if not isinstance(x, int) else [0 for _ in range(15)])
del first_category_embeddings['_merge']
first_category_embeddings.to_csv(os.path.join(raw_input_dir, "first_category_embeddings.tsv"), sep='\t', index=False, header=False)

