import pickle5 as pickle
from scipy.sparse import load_npz
from spotify_modelling import *
import time


start = time.perf_counter()

reduced_percentage = 6
n_playlists = 100
n_recs = 10

# lightfm parameters
lr = 0.05
epochs = 25
n_comp = 30
loss = 'warp'
learning_schedule = 'adagrad'


file_all_songs = "allSongs.pickle"  # name for song list pickle file

file_MF_songs = f"songlist_for_MF_reduced_{reduced_percentage}.pickle"  # input songs for MF
file_leftout_songs = f"leftout_{reduced_percentage}.pickle"  # name for list of left out songs

file_mx_reduced = f'sparse_matrix_reduced_{reduced_percentage}.npz'  # name for MF input sparse matrix
file_songmap_reduced = f'song_map_{reduced_percentage}.pickle'  # reduced song_map

file_songmap = "song_map.pickle"  # full song_map
file_mx_full = "sparseMatrix_full.npz"  # name for full matrix (used in EDA)

path = '../preprocessing/'

#%% Load sparse matrix
print("Loading files...")
mx = load_npz(path+file_mx_reduced)

# read playlists for MF (without left out songs)
with open(path+file_MF_songs, 'rb') as f:
    playlists = pickle.load(f)

# read leftout songs
with open(path+file_leftout_songs, 'rb') as f:
    leftout_songs = pickle.load(f)

# Read song_map to recreate
with open(path+file_songmap_reduced, 'rb') as f:
    songmap_reduced = pickle.load(f)



recommendations = lightfm_model(mx, playlists, songmap_reduced, lr=lr, comp=n_comp, ep=epochs, sched=learning_schedule,
                                loss=loss, n_recommendations=n_recs, n_playlists=n_playlists)

hit_rate = mean_hit_rate(recommendations, leftout_songs)
print(hit_rate)

end = time.perf_counter()
print(f"\n\nTOTAL RUN DURATION: {end-start}")
