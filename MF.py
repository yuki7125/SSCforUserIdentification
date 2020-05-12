# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm_notebook

# Matrix Factorization Implementation using Alternating Least Squares
# Includes bias terms and regularization terms for both accounts and items
# Reference: https://blog.insightdatascience.com/explicit-matrix-factorization-als-sgd-and-all-that-jazz-b00e4d9b21ea

class ML_MatrixFactorization:

    def __init__(self, k, lmd_u=0, lmd_v=0):
        # Hyperparameters for the MF
        self.k = k
        self.lmd_u = lmd_u
        self.lmd_v = lmd_v

        # Variables needed for update during fitting
        self.u = None
        self.v = None

    def fit(self, X, epochs=50):
        # X a df that is sorted (ascending) by 'new_userId' and 'new_movieId',
        # and both ids start from 0 and increase in 1's with no gap in between

        num_users = len(np.unique(X.new_userId))
        num_movies = len(np.unique(X.new_movieId))
        self.u = np.random.rand(num_users, self.k)
        self.v = np.random.rand(num_movies, self.k)

        u_bias = np.random.rand(num_users, 1)
        v_bias = np.random.rand(num_movies, 1)

        u_bias_base = np.array([1]*num_users).reshape(-1,1)
        v_bias_base = np.array([1]*num_movies).reshape(-1,1)

        # Progress Bar
        pbar = tqdm_notebook(total=epochs)

        # Fitting the MF Model
        for epoch in range(epochs):
            print('Epoch {}'.format(epoch))

            # Update User Matrix
            u_tilde = np.hstack((self.u.copy(), u_bias))
            v_tilde = np.hstack((self.v.copy(), v_bias_base))

            for ind, i in enumerate(np.unique(X.new_userId)):
                first_term = X[X.new_userId==i]['rating'].values - v_bias[X[X.new_userId==i]['new_movieId']].flatten()
                # Gets vectors of v which correspond to the movieId's that the user rated
                V = v_tilde[X[X.new_userId==i]['new_movieId']].copy()
                last_term = np.linalg.pinv(V.T @ V + self.lmd_u)
                u_tilde[ind] = first_term @ V @ last_term

            self.u = u_tilde[:,:-1].copy()
            u_bias = u_tilde[:,-1].reshape(-1,1)

            # Update Item Matrix
            u_tilde = np.hstack((self.u.copy(), u_bias_base))
            v_tilde = np.hstack((self.v.copy(), v_bias))

            for ind, j in enumerate(np.unique(X.new_movieId)):
                first_term = X[X.new_movieId==j]['rating'].values - u_bias[X[X.new_movieId==j]['new_userId']].flatten()
                U = u_tilde[X[X.new_movieId==j]['new_userId']].copy()
                last_term = np.linalg.pinv(U.T @ U + self.lmd_v)
                v_tilde[ind] = first_term @ U @ last_term

            self.v = v_tilde[:, :-1].copy()
            v_bias = v_tilde[:,-1].reshape(-1,1)

            # Calculate Objective Function (without penalty terms for simplicity)
            obj_func = 0
            for ind, i in enumerate(np.unique(X.new_userId)):
                V = self.v[X[X.new_userId==i]['new_movieId']]
                obj_func += 1/2 * np.sum(X[X.new_userId==i]['rating'].values- self.u[ind] @ V.T - u_bias[ind] - v_bias[X[X.new_userId==i]['new_movieId']].flatten())**2
            print('Objective Value: {}'.format(obj_func))

            pbar.update(1)
