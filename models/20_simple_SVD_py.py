from scipy import sparse
from scipy.sparse.linalg import svds
import pickle
import numpy as np
import pandas as pd


# Load sparse matrix
mx = sparse.load_npz("../data_preprocessing_python/sparse_matrix_reduced_2.npz")

# read playlists
with open("../data_preprocessing_python/allSongs_reduced_2.pickle", 'rb') as f:
    playlists = pickle.load(f)

# Read song_map to recreate
with open("../data_preprocessing_python/song_map_2.pickle", 'rb') as f:
    smap = pickle.load(f)

# inverse song_map to easily access songs by index
inv_map = {v: k for k, v in smap.items()}


# SVD

# Decompose to matrices from sparse matrix. "svds" is a sparse option for standard "svd".
# Still have to figure out how number of eigenvalues (k) matters to predictions
U, sigma, V = svds(mx, k = 15)

# diagonal matrix of eigenvalues, needed for reconstruction
sigma = np.diag(sigma)

# reconstruct original matrix by producing dot product U . Sigma . V
recommendations = np.dot(np.dot(U, sigma), V)


# Reconstructed matrix in dataframe format. Column names set to song names
# !!!! The values are not probabilities yet. Need to normalize the values to scale 0-1
preds_df = pd.DataFrame(recommendations, columns = list(smap.keys()))
preds_df.head()


# Show top 5 recommendations. Index means a specific playlist you want to see the recommendations for
idx = 1091 # index for specific playlist

# Top 5 recommendations, songs already present in playlist are removed
print("Top 5 Recommendations")
print([s for s in preds_df.iloc[idx].sort_values(ascending=False).index if s not in playlists[idx]][:5], end="\n\n\n")

# Show 20 first songs from playlist to compare the recommendations. Use same index as previous cell
print(playlists[idx][:20])
