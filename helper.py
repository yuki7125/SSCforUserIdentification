# -*- coding: utf-8 -*-

import random
import numpy as np
import pandas as pd

### Helper functions to create new IDs

# Create new_userId, by reordering from 0 increasing by 1's
def create_new_userid(df):
    new_id_list = []
    for new_id, old_id in enumerate(np.unique(df.userId)):
        new_id_list.extend([new_id] * len(df[df['userId']==old_id]))
    return new_id_list


# Create new_movieId, by reordering from 0 increasing by 1's
def create_new_movieid(df):
    new_id_list = []
    new_id_dict = {}
    for new_id, old_id in enumerate(np.unique(df.movieId)):
        new_id_dict[old_id] = new_id
    for old_id in list(df.movieId):
        new_id_list.append(new_id_dict[old_id])
    return new_id_list


### Helper functions to create artificial 2-user dataset

# Make pairs of users randomly
def output_paired_users(df):

    def choose_user(users):
        user_idx = random.randrange(0, len(users))
        return users.pop(user_idx)

    users = list(np.unique(df.new_userId))

    user_pairs = []

    while users:
        user1 = choose_user(users)
        user2 = choose_user(users)
        user_pair = [user1, user2]
        user_pairs.append(user_pair)

    return np.array(user_pairs)


# Get corresponding movies of paired users
def output_corresponding_movies(df, user_pairs):
    account_movies_list = []
    for user1, user2 in user_pairs:
        movie_list = []
        movie_list.extend(df[df.new_userId == user1]['new_movieId'])
        movie_list.extend(df[df.new_userId == user2]['new_movieId'])
        account_movies_list.append(movie_list)

    return np.array(account_movies_list)


# Get movie_id's of each cluster of a single account
def collect_movies_in_clusters(account_movies, pred_labels):
    movies_0 = []
    movies_1 = []

    for movie_id, label in zip(np.unique(account_movies), pred_labels):
        if label==0:
            movies_0.append(movie_id)
        else:
            movies_1.append(movie_id)

    return movies_0, movies_1


# Get all genre's corresponding to movie_id's in a list
def get_genres(movies_list):
    movies_df = pd.read_csv('movielens/movies.csv')
    genre_list = []
    for genre in movies_df[movies_df.movieId.isin(movies_list)]['genres'].values:
        genre_list.extend(genre.split('|'))
    return genre_list


# Calculate Ratio Difference between clusters
def calculate_cluster_ratio_difference(keys_0, keys_1, values_0, values_1):
    ratio_diff = 0
    for key in list(set(keys_0) | set(keys_1)):
        if (key in keys_0) & (key in keys_1):
            ratio_diff += np.abs(values_0[keys_0.index(key)]- values_1[keys_1.index(key)])
        elif key in keys_0:
            ratio_diff += values_0[keys_0.index(key)]
        else:
            ratio_diff += values_1[keys_1.index(key)]

    return ratio_diff
