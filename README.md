# K-Means clustering with the distance matrix

An undirected graph data used for this project. It represents connected blogs with labeled two classes. In this project, K - Means used for clustering this data and calculation has been done for [F-Measure](https://en.wikipedia.org/wiki/F1_score) and [Purity](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html). 

1. The data pre-processed for producing connection matrix and then similarity matrix producing with similarity functions. In this particular project, the [Manhattan Distance](https://en.wikipedia.org/wiki/Taxicab_geometry) has been used for similarities. 

---
*Example Connection Matrix*

 | 0        | 1           | 2  |3           | 4  |
| ------------- |:-------------:| -----:|:-------------:| -----:|
| **1**      | 0 | 1 | 1 | 0 |
| **2**      | 1      |   1 | 0 | 0 |
| **3** | 0      |    0 | 0 | 1 |
| **4** | 0      |    1 | 0 | 1 |

*Example Distance(Similarity) Matrix*

 | 0        | 1           | 2  |3           | 4  |
| ------------- |:-------------:| -----:|:-------------:| -----:|
| **1**     | 1.000   | 0.205 | 0.000 | 0.000 |
| **2**     | 0.205   | 1.000 | 0.333 | 0.222 |
| **3**     | 0.000   | 0.333 | 1.000 | 0.721 |
| **4**     | 0.000   | 0.222 | 0.721 | 1.000 |

---

2. __K__ number random centroids with __N__ random attributes (i.e., the similarity with respect to each node) added. 
3. K-Means performed on the similarity matrix
  - Which centroid each of the __N__ nodes is closest to found (distance each node with respect to each centroid)
  - Centroids updated after each clustering (average of each attribute --rows-- for the cluster in the matrix calculated)
4. [Precision, Recall](https://en.wikipedia.org/wiki/Precision_and_recall) and Purity (with Entropy) calculated. 

---
Graph-1: Label x N
![Clustering1 ](https://github.com/dauut/kmeans-manhattan-jaccard-distance/blob/master/graphs/Clustering_LabelxN.png "Label x N")

Graph-2: N x N 
![Clustering - 2](https://github.com/dauut/kmeans-manhattan-jaccard-distance/blob/master/graphs/Clustering_NxN.png "N x N")

---
