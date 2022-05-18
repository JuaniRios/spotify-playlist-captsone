import pickle5 as pickle
from scipy.sparse import load_npz
from modeling.modeling_functions import simple_svd, mean_hit_rate
import time
from decouple import config
import os
os.chdir(config("PROJECT_PATH"))

start = time.perf_counter()

n_playlists = 100
n_recs = 10
reduced_percentage = int(config("REDUCED_PERCENTAGE"))

file_all_songs = config("FILE_ALL_SONGS")
file_songmap = config("FILE_SONGMAP")
file_mx_full = config("FILE_MX_FULL")

file_MF_songs = config("FILE_MF_SONGS").format(reduced_percentage=reduced_percentage)
file_leftout_songs = config("FILE_LEFTOUT_SONGS").format(reduced_percentage=reduced_percentage)

file_songmap_reduced = config("FILE_SONGMAP_REDUCED").format(reduced_percentage=reduced_percentage)
file_mx_reduced = config("FILE_MX_REDUCED").format(reduced_percentage=reduced_percentage)

#%% Create simple SVD Model

# load sparse matrix
print("Loading files...")
mx = load_npz(file_mx_reduced)

# read playlists for MF (without left out songs)
with open(file_MF_songs, 'rb') as f:
    playlists = pickle.load(f)

# read leftout songs
with open(file_leftout_songs, 'rb') as f:
    leftout_songs = pickle.load(f)

# Read song_map to recreate
with open(file_songmap_reduced, 'rb') as f:
    songmap_reduced = pickle.load(f)

recommendations = simple_svd(mx, playlists, songmap_reduced, n_recommendations=n_recs, n_playlists=n_playlists)

hit_rate = mean_hit_rate(recommendations, leftout_songs)
print(hit_rate)
# 100 playlists: 0.69; 1000 playlists: 0.937; 10 000 playlists: 0.8841, 100 000: 0.8723 in top 10 recommendations

# after data set compression:
# 1000 playlists, 2% reduced percentage, leftout_cutoff .9, min_songs: 5, min_song_count = 5: 0.14 -> too few training samples?
# 1000 playlists, 3% reduced percentage, leftout_cutoff .75, min_songs 40, min_song_count = 5: .548
# 1000 playlists, 3% reduced percentage, leftout_cutoff .9, min_songs 40, min_song_count = 5: .205
# 1000 playlists, 2% reduced percentage, leftout_cutoff .7, min_songs 40, min_song_count = 1: 0.66
# 100 playlists, 2% reduced percentage, leftout_cutoff .9, min_songs 1, min_song_count = 1: 0.66

# RESULTS NOT REPRODUCIBLE ANYMORE!!!

end = time.perf_counter()
print(f"\n\nTOTAL RUN DURATION: {end-start}")
