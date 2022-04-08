# This file includes all functions needed for the modelling process.

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

def simple_svd(matrix, playlists, songmap, n_recommendations = 10, n_playlists = 100):
    '''
    Input:
        - matrix: coordinate sparse matrix
        - playlists: all playlists to consider; must correspond to sparse matrix
        - songmap: songmap mapping each song to its index; must correspond to playlists
        - n_recommendations: How many recommendations would you like to consider? Default: 10
        - n_playlists: How many playlists do you want to create recommendations for? Default: 100
    This function creates the recommendations for given sparse matrix.
    Output: List of lists containing recommendations for each playlist and each song.
    '''

    # Decompose to matrices from sparse matrix. "svds" is a sparse option for standard "svd".
    # We still have to figure out how number of eigenvalues (k) matters to predictions

    # To avoid ValueError: matrix type must be 'f', 'd', 'F', or 'D'
    # upcast matrix to fload or double.
    matrix = matrix.asfptype()
    # create matrices
    print("Decomposing matrix...")
    U, sigma, V = svds(matrix, k=15)

    # diagonal matrix of eigenvalues, needed for reconstruction
    sigma = np.diag(sigma)

    print("Calculating recommendations...")
    # reconstruct original matrix by producing dot product U . Sigma . V
    recommendations = np.dot(np.dot(U, sigma), V)
    # make above calculation more memory efficient: use np array of arrays
    # / only save recommendations above a certain probability and use sparse matrix again...

    # Reconstruct Matrix for Recommendations
    # Reconstructed matrix in dataframe format. Column names set to song names
    # !!!! The values are not probabilities yet. Need to normalize the values to scale 0-1
    print("Reconstructing matrix...")
    preds_df = pd.DataFrame(recommendations, columns=list(songmap.keys()))
    #TODO: This process consumes a lot of time and processing power. Optimize! Ideas: np.arrays
    preds_df.head()

    # Access Recommendations by Playlist Index
    # Show top n_recommendations recommendations.
    # Index means a specific playlist you want to see the recommendations for

    results = []  # store recommendations for each playlist
    for idx in range(len(playlists[:n_playlists])):
        # go through each playlist until defined limit n_playlists
        print(f"{round(idx / n_playlists * 100, 2)}% done.")
        # Top n recommendations, songs already present in playlist are removed
        # Access top n songs indices with highest probability by index of playlist. Use these indices to access songs
        # in playlist.
        recs = [s for s in preds_df.iloc[idx].sort_values(ascending=False).index if s not in playlists[idx]]
        results.append(recs[:n_recommendations])
        # print(SVD_recs, end="\n\n\n")

        # Show 20 first songs from playlist to compare the recommendations. Use same index as previous cell
        #print(f"First 10 recommendations for playlist nr. {idx}: {recommendations[idx][:20]}")

    return results

def mean_hit_rate(playlists, recommendations, leftout_songs, n_playlists = 100):
    '''
    Input:
        - playlists: playlists for which the recommendations were made and were input to the simple_svd()
        - recommendations: list of recommendations for each playlist which was the output of simple_svd()
        - leftout_songs: list of songs which were removed for evaluation purposes
        - n_playlists: number of playlists for which recommendations were made (== len(playlists)? Have to check -> if so, you can remove this parameter!)
    This function evaluates the results of the recommendations by calculating the hit rate.
    Output: average hit rate over all playlists
    '''
    hit_rates = [] # store hit rates for each playlist
    for idx in range(len(playlists[:n_playlists])):
        # Calculate Hit Rate for each playlist
        # sum up all songs in leftout_songs which are in our top recommendations for each playlist
        hit_rates.append(sum([song in leftout_songs[idx] for song in recommendations[idx]]))
    print("Mean Hit Rate:")
    # calculate mean
    mean_hit_rate = sum(hit_rates) / len(hit_rates)
    return mean_hit_rate