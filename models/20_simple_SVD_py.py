from scipy import sparse
from scipy.sparse.linalg import svds
import pickle
import numpy as np
import pandas as pd

#%% Load sparse matrix
mx = sparse.load_npz("../data_preprocessing_python/sparse_matrix_reduced_2.npz")

# read playlists
with open("../data_preprocessing_python/allSongs_reduced_2.pickle", 'rb') as f:
    playlists = pickle.load(f)

# read leftout songs
with open("../data_preprocessing_python/leftout.pickle", 'rb') as f:
    leftout_songs = pickle.load(f)

# Read song_map to recreate
with open("../data_preprocessing_python/song_map_2.pickle", 'rb') as f:
    smap = pickle.load(f)

# inverse song_map to easily access songs by index
inv_map = {v: k for k, v in smap.items()}


#%% SVD

# Decompose to matrices from sparse matrix. "svds" is a sparse option for standard "svd".
# Still have to figure out how number of eigenvalues (k) matters to predictions

# To avoid ValueError: matrix type must be 'f', 'd', 'F', or 'D'
# upcast matrix to fload or double.
mx = mx.asfptype()
# create matrices
U, sigma, V = svds(mx, k=15)

# diagonal matrix of eigenvalues, needed for reconstruction
sigma = np.diag(sigma)

# reconstruct original matrix by producing dot product U . Sigma . V
recommendations = np.dot(np.dot(U, sigma), V)

#%% Reconstruct Matrix for Recommendations
# Reconstructed matrix in dataframe format. Column names set to song names
# !!!! The values are not probabilities yet. Need to normalize the values to scale 0-1
preds_df = pd.DataFrame(recommendations, columns=list(smap.keys()))
preds_df.head()

#%% Access Recommendations by Playlist Index
# Show top 10 recommendations. Index means a specific playlist you want to see the recommendations for
# track hitrate
hr = []
for i in range(len(playlists)):
    idx = i  # index for specific playlist
    print(i/len(playlists))
    # Top 20 recommendations, songs already present in playlist are removed
    #print("Top 20 Recommendations")
    SVD_recs = [s for s in preds_df.iloc[idx].sort_values(ascending=False).index if s not in playlists[idx]][:20]
    #print(SVD_recs, end="\n\n\n")

    # Calculate Hit Rate
    hr.append(sum([r in leftout_songs[idx] for r in SVD_recs]))
    #print("Hit Rate:")
    #print(hr[i])

    # Show 20 first songs from playlist to compare the recommendations. Use same index as previous cell
    #print(playlists[idx][:20])
print("Mean Hit Rate:")
print(sum(hr)/len(hr))
