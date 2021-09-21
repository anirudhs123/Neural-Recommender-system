# Neural-Reccomender-system

Built a whole pipeline of recommender systems comprising of Popularity based recommender, KNN similarity based Clustering recommender, Item-Item association based recommender, Bi-Partite graph based association recommender, Neural Graph based Collaborative Filtering and Neural Embedding based Collaborative filtering. These mdoels were built to work on three different datasets namely Amazon Book Dataset, Movie Lens Dataset and a Supermarket user-item interaction dataset.

1. The pipeline includes a Co-clustering algorithm based recommender built using the Scikit-Surprise package. There were options to use either cosine similarity or Pearson similarity based collaborative filtering recommender, both of which were built using the Turicreate package. 
2. The piepline also includes a simple popularity based recommender which takes into account the transaction frequency of each item. We then predict the Top-k in terms of frequency of occurrence as recommendations.

<p align="center">
   <img src="../gh-pages/assets/images/gesture_sample.jpg" width=400 height=300>
</p>



## Key Contributions

1. Machine algorithms used for traditional collaborative filtering is not very efficient in handling sparse data.The Neural Embedding based recommender was built specially for handling sparse input data. We observe from the results that Matrix Factorization methods are being outperformed by the Embedding based Recommenders.
2. The Bi-Partite graph based association recommender was used for considering co-occurrences among items and to consider higher order proximities among the items. The Bi-partite graph based recommenders were built using Louivan community partition and Apriory algorithms.
3. Item-Item association recommender was built based on Apriori algorithm

## People

This work has been developed during a internship as a Data Scientist Intern by [Anirudh Sriram](https://github.com/anirudhs123) from Indian Institute of Technology, Madras. Ask us your questions at [anirudhsriram30799@gmail.com](mailto:anirudhsriram30799@gmail.com).
