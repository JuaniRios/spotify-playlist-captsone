import os
from decouple import config
os.chdir(config("PROJECT_PATH"))
import pickle5 as pickle
from scipy.sparse import load_npz
from modeling.modeling_functions import *
import time



reduced_percentage = int(config("REDUCED_PERCENTAGE"))
n_playlists = 100
n_recs = 10

# lightfm parameters
lr = 0.05
epochs = 25
n_comp = 30
loss = 'warp'
learning_schedule = 'adagrad'

file_all_songs = config("FILE_ALL_SONGS")
file_songmap = config("FILE_SONGMAP")
file_mx_full = config("FILE_MX_FULL")

file_MF_songs = config("FILE_MF_SONGS").format(reduced_percentage=reduced_percentage)
file_leftout_songs = config("FILE_LEFTOUT_SONGS").format(reduced_percentage=reduced_percentage)

file_songmap_reduced = config("FILE_SONGMAP_REDUCED").format(reduced_percentage=reduced_percentage)
file_mx_reduced = config("FILE_MX_REDUCED").format(reduced_percentage=reduced_percentage)

# %% Create model
start = time.perf_counter()
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

recommendations = lightfm_model(mx, playlists, songmap_reduced, lr=lr, comp=n_comp, ep=epochs, sched=learning_schedule,
                                loss=loss, n_recommendations=n_recs, n_playlists=n_playlists)

# save model for later use
with open("lightFM_model.pickle", "wb+") as f:
    pickle.dump(recommendations, f, protocol=pickle.HIGHEST_PROTOCOL)

end = time.perf_counter()
print(f"\n\nTOTAL RUN DURATION: {end - start}")

#%% Check performance
with open("lightFM_model.pickle", "rb") as f:
    recommendations = pickle.load(f)

hit_rate = mean_hit_rate(recommendations, leftout_songs)
print(hit_rate)


