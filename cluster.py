# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from tqdm import tqdm_notebook

# Function for clustering and calculating accuracy for Kmeans, Spectral Clustering, and Sparse Subspace Clustering

def cluster_labels_with_acc(v, df, user_pairs_arr, account_movies_arr, n_clusters, algo):
    acc_list = []
    all_pred_labels = []

    pbar = tqdm_notebook(total=len(user_pairs_arr))

    for user_pair, account_movies in zip(user_pairs_arr, account_movies_arr):

        # Add ratings of the users to each row of vs
        df_user = pd.DataFrame({'new_userId':user_pair}).merge(df)
        vs = np.hstack((v[account_movies], df_user.rating.values.reshape(-1,1)))


        # Cluster using Kmeans
        if algo == 'kmeans':
            pred_labels = KMeans(n_clusters=n_clusters, random_state=0).fit(vs).labels_

        # Cluster using Spectral Clustering
        elif algo == 'spectral':
            state = 0
            while(1):
                try:
                    pred_labels = SpectralClustering(n_clusters=n_clusters, eigen_solver='lobpcg', random_state=state).fit(vs).labels_
                    break

                except:
                    state+=1
                    continue

        # Cluster using Sparse Subspace Clustering
        elif algo == 'ssc':
            c_mat = np.zeros([len(vs), len(vs)])

            pbar2 = tqdm_notebook(total=len(vs))
            for i in range(len(vs)):

                lmda = 10
                c_k = np.random.random(len(vs)-1)
                c_k_prev = c_k.copy()
                t_k = 1
                Y_not_i = np.delete(vs, i, axis=0).T

                # epsilon must be low to ensure sparsity in target c_k
                epsilon = 0.1

                eigval, eigvect = np.linalg.eig(Y_not_i.T @ Y_not_i)
                L = np.max(eigval)

                start = 0

                # Accelerated Proximal Gradient for Basis Pursuit Problem
                while (np.linalg.norm(c_k_prev-c_k) > epsilon) | (start==0):
                    t_k1 = (1+np.sqrt(1+4*(t_k**2)))/2
                    beta_k1 = (t_k-1)/t_k1
                    t_k = t_k1.copy()

                    p_k1 = c_k + beta_k1 * (c_k-c_k_prev)
                    w_k1 = p_k1 - 1/L * Y_not_i.T @ (Y_not_i @ p_k1 - vs[i])
                    c_k_prev = c_k.copy()
                    c_k = w_k1/np.abs(w_k1) * np.maximum(np.abs(w_k1)-lmda/L, np.zeros(len(w_k1)))

                    #obj = 1/2 * np.linalg.norm(vs[i]-Y_not_i @ c_k) + lmda * np.linalg.norm(c_k, 1)
                    #print(obj)

                    start += 1

                pbar2.update(1)
                c_mat[i] = np.insert(c_k.copy(), i, 0)

            C_tilde = np.abs(c_mat) + np.abs(c_mat.T)

            state = 0
            while(1):
                try:
                    pred_labels = SpectralClustering(n_clusters=n_clusters, random_state=state,
                                                     eigen_solver='amg',affinity='precomputed').fit(C_tilde).labels_
                    break

                except:
                    state+=1
                    continue


        else:
            sys.exit('Algorithm does not exist')

        all_pred_labels.append(pred_labels)

        # Get predicted and true labels for each user
        user1_pred_labels = pred_labels[:len(df[df.new_userId==user_pair[0]]['new_movieId'])]
        user2_pred_labels = pred_labels[len(df[df.new_userId==user_pair[0]]['new_movieId']):]
        user1_true_labels = [0 for i in range(len(df[df.new_userId==user_pair[0]]['new_movieId']))]
        user2_true_labels = [1 for i in range(len(df[df.new_userId==user_pair[1]]['new_movieId']))]

        # Get labels that were correctly predicted for each user
        user1_eval = [i for i, j in zip(user1_pred_labels, user1_true_labels) if i == j]
        user2_eval = [i for i, j in zip(user2_pred_labels, user2_true_labels) if i == j]

        # Record the accuracy (Will always be above 0.5)
        correct_perc = (len(user1_eval) + len(user2_eval))/len(pred_labels)
        if correct_perc > 0.5:
            acc_list.append(correct_perc)
        else:
            acc_list.append(1-correct_perc)

        pbar.update(1)

    return acc_list, all_pred_labels
