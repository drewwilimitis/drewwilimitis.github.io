---
title: Projects
permalink: /projects/
layout: page
excerpt: 
comments: 
---

# Manifold Learning: Introduction and Foundational Algorithms in Python <br>
After I was unable to find a thorough investigation of manifold learning methods, I decided to try and learn from a wide variety of sources and then attempt to synthesize an in-depth explanation of the core algorithms, along with their implementations within sklearn.manifold. <br>

Note: the notebooks are best viewed through the jupyter nbviewer links below. [github](https://github.com/drewwilimitis/Manifold-Learning)

![KleinDual](https://user-images.githubusercontent.com/16658498/55211718-3e737900-51bb-11e9-8a4a-37ecf83c9ab0.gif)


## Contents


* **Introduction:** [nbviewer](https://nbviewer.jupyter.org/github/drewwilimitis/Manifold-Learning/blob/master/Manifold_Learning_Intro.ipynb)
    - Overview of manifolds and the basic topology of data
    - Statistical learning and instrinsic dimensionality
    - The manifold hypothesis 

* **Chapter 1: Multidimensional Scaling** [nbviewer](https://nbviewer.jupyter.org/github/drewwilimitis/Manifold-Learning/blob/master/Multidimensional_Scaling.ipynb)
    - Classical, metric, and non-metric MDS algorithms
    - Example applications to quantitative psychology and social science
    
* **Chapter 2: ISOMAP** [nbviewer](https://nbviewer.jupyter.org/github/drewwilimitis/Manifold-Learning/blob/master/Isomap.ipynb)
    - Geodesic distances and the isometric mapping algorithm
    - Implementation details and applications with facial images and coil-100 object images 

* **Chapter 3: Local Linear Embedding** [nbviewer](https://nbviewer.jupyter.org/github/drewwilimitis/Manifold-Learning/blob/master/Locally_Linear_Embedding.ipynb)
    - Locally linear reconstructions and optimization problems
    - Example applications with image data
    
* **Chapter 4: Laplacian Eigenmaps/Spectral Embedding** [nbviewer](https://nbviewer.jupyter.org/github/drewwilimitis/Manifold-Learning/blob/master/Laplacian-Eigenmaps.ipynb)
    - From the general to the discrete Laplacian operators
    - Visualizing spectral embedding with the networkx library
    - Spectral embedding with NLTK and the Brown text corpus

<br>

-----
-----
<br>

# hyperbolic-learning

**It has been recently established that many real-world networks have a latent geometric structure that resembles negatively curved hyperbolic spaces. Therefore, complex networks, and particularly the hierarchical relationships often found within, can often be more accurately represented by embedding graphs in hyperbolic geometry, rather than flat Euclidean space.**

**The goal of this project is to provide Python implementations for a few recently published algorithms that leverage hyperbolic geometry for machine learning and network analysis. Several examples are given with real-world datasets, however; the time complexity is far from optimized and this repository is primarily for research purposes - specifically investigating how to integrate downstream supervised learning methods with hyperbolic embeddings.**

![IllinformedHalfAnemone-size_restricted](https://user-images.githubusercontent.com/16658498/67390992-564d8880-f563-11e9-95af-a959085e72c1.gif)

Note: the notebooks are best viewed through the jupyter nbviewer links below. [github](https://github.com/drewwilimitis/hyperbolic-learning)

## Contents

### Models
* __Poincaré Embeddings__: 
  - Mostly an exploration of the hyperbolic embedding approach used in __[1]__.
  - Available implementation in the `gensim` library and a PyTorch version released by the authors [here](https://github.com/facebookresearch/poincare-embeddings).
  
* __Hyperbolic Multidimensional Scaling__: [nbviewer](https://nbviewer.jupyter.org/github/drewwilimitis/hyperbolic-learning/blob/master/hyperbolic_mds/hyperbolic_mds_results.ipynb)
  - Finds embedding in Poincaré disk with hyperbolic distances that preserve input dissimilarities __[2]__.
  
* __K-Means Clustering in the Hyperboloid Model__: [nbviewer](https://nbviewer.jupyter.org/github/drewwilimitis/hyperbolic-learning/blob/master/hyperbolic_kmeans/hyperbolic_kmeans_results.ipynb)
  - Optimization approach using Frechet means to define a centroid/center of mass in hyperbolic space __[3, 4]__.


  ![mammals_kmeans](https://user-images.githubusercontent.com/16658498/69854762-7f8ec200-124f-11ea-9cab-719efc965aa2.png)

  
* __Hyperbolic Support Vector Machine__ - [nbviewer](https://nbviewer.jupyter.org/github/drewwilimitis/hyperbolic-learning/blob/master/hyperbolic_svm/hyperbolic_svm_results.ipynb)
  - Linear hyperbolic SVC based on the max-margin optimization problem in hyperbolic geometry __[5]__.
  - Uses projected gradient descent to define decision boundary and predict classifications.


![hsvm_decision_boundaries](https://user-images.githubusercontent.com/16658498/69854243-39852e80-124e-11ea-97aa-d8d34ba107d8.png)

  
* __Embedding Graphs in Lorentzian Spacetime__ -  [nbviewer](https://nbviewer.jupyter.org/github/drewwilimitis/hyperbolic-learning/blob/master/spacetime_graph_embeddings/spacetime_embedding_results.ipynb)
  - An algorithm based on notions of causality in the Minkowski spacetime formulation of special relativity __[6]__.
  - Used to embed directed acyclic graphs where nodes are represented by space-like and time-like coordinates. 

![hep-th_citation_network](https://user-images.githubusercontent.com/16658498/65956193-6fa16000-e40f-11e9-935b-a518a77b6525.png)

### Datasets
- Zachary Karate Club Network
- WordNet
- Enron Email Corpus
- Polbooks Network
- arXiv Citation Network
- Synthetic generated data (sklearn.make_datasets, networkx.generators, etc.)

### Dependencies
- Models are designed based on the sklearn estimator API (`sklearn` generally used only in rare, non-essential cases)
- `Networkx` is used to generate & display graphs

### References

__[1]__ Nickel, Kiela. "Poincaré embeddings for learning hierarchical representations" (2017). [arXiv](https://arxiv.org/pdf/1705.08039.pdf).<br>
__[2]__ A. Cvetkovski and M. Crovella. Multidimensional scaling in the Poincaré disk. arXiv:1105.5332, 2011.<br>
__[3]__ "Learning graph-structured data using Poincaré embeddings and Riemannian K-means algorithms". Hatem Hajri, Hadi Zaatiti, Georges Hebrail (2019) [arXiv](https://arxiv.org/abs/1907.01662).<br>
__[4]__ Wilson, Benjamin R. and Matthias Leimeister. “Gradient descent in hyperbolic space.” (2018).<br>
__[5]__ "Large-margin classification in hyperbolic space". Cho, H., Demeo, B., Peng, J., Berger, B. CoRR abs/1806.00437 (2018).<br>
__[6]__ Clough JR, Evans TS (2017) Embedding graphs in Lorentzian spacetime. PLoS ONE 12(11):e0187301. https://doi.org/10.1371/journal.pone.0187301.

-----
