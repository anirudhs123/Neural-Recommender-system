import pandas as pd
#!pip install turicreate
import turicreate
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances 



u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']

# Reading the movie lens dataset
# Change the path to the data accordingly
users = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Recommender systems/ml-100k/ml-100k/u.user', sep='|', names=u_cols,encoding='latin-1')

# reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Recommender systems/ml-100k/ml-100k/u.data', sep='\t', names=r_cols,encoding='latin-1')

# reading items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Recommender systems/ml-100k/ml-100k/u.item', sep='|', names=i_cols,
encoding='latin-1')

#Ratings cols
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_train = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Recommender systems/ml-100k/ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Recommender systems/ml-100k/ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')
ratings_train.shape, ratings_test.shape

# Distinct Number of items and users
n_users = ratings.user_id.unique().shape[0]
n_items = ratings.movie_id.unique().shape[0]

data_matrix = np.zeros((n_users, n_items))

for t in range(len(ratings)):
  i=ratings['user_id'].values[t]-1
  j=ratings['movie_id'].values[t]-1
  rat=ratings['rating'].values[t]

  data_matrix[i,j]=rat


user_similarity = pairwise_distances(data_matrix, metric='cosine')
item_similarity = pairwise_distances(data_matrix.T, metric='cosine')


def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #We use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

#Cosine similairty based Reco
user_prediction = predict(data_matrix, user_similarity, type='user')
item_prediction = predict(data_matrix, item_similarity, type='item')

#Turicreate Reccomender system
train_data = turicreate.SFrame(ratings_train)
test_data = turicreate.SFrame(ratings_test)

#Popularity based Reco
popularity_model = turicreate.popularity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')

popularity_recomm = popularity_model.recommend(users=[1,2,3,4,5],k=5)
popularity_recomm.print_rows(num_rows=25)

#Training the model
# Item similairty based Reco
item_sim_model = turicreate.item_similarity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating', similarity_type='cosine')

#Making recommendations
item_sim_recomm = item_sim_model.recommend(users=[1,2,3,4,5],k=5)
item_sim_recomm.print_rows(num_rows=25)

