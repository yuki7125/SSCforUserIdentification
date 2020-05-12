# Sparse Subspace Clustering for User Identification
Standard recommendation methods rely on the assumption that each account is used by a single user. In reality though, multiple users may be sharing one account. In these cases, recommendations may lean toward more dominant users, or generalize to the combined weak preferences of multiple users. In addition, with shared accounts, services may be losing potential revenue. To solve these issues, the best way is to distinguish the users sharing the account. This is known to be a chicken and egg problem, because if we know the number of users sharing the account beforehand, it is relatively easy to allocate users to the items of their preferences, and if we know the pairings of users and items beforehand, it is trivial to discover the number of users using the account. Since this problem setting is analogous to the subspace clustering setting, where the objective is to find the number of subspaces, and the data points belonging to each subspace, we apply subspace clustering to this problem. Thus, in this project the objective is to apply subspace clustering to an account x item matrix, to distinguish the accounts which are shared by multiple users. Specifically, we apply Sparse Subspace Clustering to the MovieLens dataset, with major focus on discovering the pairings of users and movies in the multiple user setting. We show results of this experiment and discuss directions we could take for further research.

# Related Work:
This project is based off of the paper "Guess Who Rated This Movie:Identifying Users Through Subspace Clustering" by Zhang et al. [1], which attempted to apply Generalized Principal Component Analysis (GPCA), which is a subspace clustering method. Unfortunately, they were not able to produce good results with GPCA due to the presence of noise in the data. Meanwhile, Sparse Subspace Clustering (SSC) [4] is currently the state-of-art in subspace clustering and is known to be able to handle noise in the data. Therefore, here the objective is to apply SSC to a similar setting and see how SSC acts.

# Notes  
SSC_for_User_Identification.ipynb is the main file
