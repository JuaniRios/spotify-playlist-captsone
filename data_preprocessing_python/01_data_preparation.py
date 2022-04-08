#%% Imports
import os

import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
# os.chdir("C:/Users/juani/PycharmProjects/spotify-playlist-captsone/data_preprocessing_python")
os.chdir("C:/Users/netzl/OneDrive/Dokumente/Studium/Informatics Krems/4h semester/Data Science Capstone/spotify-playlist-captsone/data_preprocessing_python")
#%% Load Matrix
print(os.getcwd())
print("loading matrix from file")
small_matrix = sparse.load_npz("./sparse_matrix_reduced_2.pickle.npz")
big_matrix = sparse.load_npz("../depr/sparse_matrix_reduced_2.npz")
print("finished loading matrix")

#%% plot matrix
plt.spy(matrix)
plt.show()

#%% test
A = sparse.rand(10000,10000, density=0.00001)
M = sparse.csr_matrix(A)
plt.spy(M)
plt.show()

#%% look at output files
import pickle
reduced_percentage = 3

file_all_songs = "allSongs.pickle"  # name for song list pickle file

file_MF_songs = f"songlist_for_MF_reduced_{reduced_percentage}.pickle"  # input songs for MF
file_leftout_songs = f"leftout_{reduced_percentage}.pickle"  # name for list of left out songs

file_mx_reduced = f'sparse_matrix_reduced_{reduced_percentage}.npz'  # name for MF input sparse matrix
file_songmap_reduced = f'song_map_{reduced_percentage}.pickle'  # reduced song_map

file_songmap = "song_map.pickle"  # full song_map
file_mx_full = "sparseMatrix_full.npz"  # name for full matrix (used in EDA)

path = '../data_preprocessing_python/'

# read playlists for MF (without left out songs)
with open(path+file_MF_songs, 'rb') as f:
    playlists = pickle.load(f)

# read leftout songs
with open(path+file_leftout_songs, 'rb') as f:
    leftout_songs = pickle.load(f)

# Read song_map to recreate
with open(path+file_songmap_reduced, 'rb') as f:
    songmap_reduced = pickle.load(f)
