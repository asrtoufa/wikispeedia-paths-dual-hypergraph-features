#### Download

Download data from:
 
http://snap.stanford.edu/data/wikispeedia.html

#### Details

- `categories.tsv` - contains the categories each article belongs to.
There is a hierarchical categorization of the articles with 3 levels, 
starting from a broad category to a more specialized one. 
In the first level there are 15 categories. 
In the other two levels, there are 102 and 28 different categories.

- `first_category_embeggings.tsv `- is generated using the 
`gretel/wikipedia_preprocessing.py` script.
The first level categories are represented as a one-hot vector of length 15.
Each article is associated with one of these vectors indicating its 
relationship with the corresponding category. 

- `article_embeddings.txt` - input file from 
https://github.com/jbcdnr/gretel-path-extrapolation
