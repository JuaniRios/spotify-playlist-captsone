import json
import os
from itertools import chain
import numpy as np
from scipy.sparse import vstack, save_npz, coo_matrix
import time
import pickle
from ordered_set import OrderedSet


#%% Environment Variables
# path = "../../ML_AI/ds_capstone/data/"  # Henrik
path = "C:/Users/netzl/Offline Documents/spotify_million_playlist_dataset/data/" # Daniel
# path = "../data/spotify_million_playlist_dataset/data/" # Juan
reduced_percentage = 2
f_name_songlist = "allSongs_reduced_2.pickle" # name for song list pickle file
f_name_mx = 'sparse_matrix_reduced_2.npz' # name for sparse matrix
f_name_song_map = "song_map_2.pickle" # name for song_map pickle (to keep the order of songs in mx)
f_name_leftout = "leftout_2.pickle" # name for list of left out songs
start = time.perf_counter()
#os.chdir("/data_preprocessing_python")

#%% Create list of playlists
def pickle_playlists(filename, data_loc, percentage):
    print(os.getcwd())
    db = []
    leftout_songs = []
    # get all files (dataset is divided)
    files = os.listdir(data_loc)
    amount_files = len(files)
    limit = int(amount_files * (percentage/100))
    reduced_files = files[:limit]
    for file in reduced_files:
        print(f"getting {file}")
        data = json.load(open(data_loc + file))
        # list of playlist are under the key "playlists"
        for row in data["playlists"]:
            leftout_cutoff = int(round(len(row) * .9, 0))  # 90% of each playlist for creating recommendation
            tracks = [song["artist_name"] + " - " + song["track_name"] for song in row["tracks"]]
            db.append(tracks[:leftout_cutoff])
            leftout_songs.append(tracks[leftout_cutoff:]) # 10% for evaluation/calculating hit rate

    with open(filename, 'wb') as f:
        pickle.dump(db, f, pickle.HIGHEST_PROTOCOL)

    with open(f_name_leftout, 'wb') as f:
        pickle.dump(leftout_songs, f, pickle.HIGHEST_PROTOCOL)

if not (os.path.isfile(f_name_songlist) & os.path.isfile(f_name_leftout)):
    print("Pickle not found, creating it now. Please wait...")
    pickle_playlists(f_name_songlist, path, reduced_percentage)
    print("Created pickled playlists")
else:
    print("Pickle found :)")

print("Reading Pickle")
with open(f_name_songlist, 'rb') as f:
    playlists = pickle.load(f)
print("Pickle reading finished")

#%% Get vectors for sparse matrix (coordinates in a playlist~songlist where a song is found)
# rows were each playlist. we eliminate rows and remove duplicates to get unique songs
print("Creating vectors for spare matrix")

song_list = list(OrderedSet(chain(*playlists))) # shuffles songs
print(len(song_list))
print(song_list[:5])

song_map = {song:i for i,song in enumerate(song_list)} # thus, random

# Save Songmap
with open(f_name_song_map, 'wb') as f:
     pickle.dump(song_map, f, pickle.HIGHEST_PROTOCOL)

data_m, row_ix, col_ix = [], [], []

for rowIdx, rw in enumerate(playlists):
     for dbSong in rw:
         row_ix.append(rowIdx)
         col_ix.append(song_map[dbSong])
data_m = [1]*len(col_ix)
print("Finished creating vectors")

#%% Create Sparse Matrix
print("Creating Sparse Matrix")
sparseMatrix = coo_matrix((data_m, (row_ix, col_ix)), shape=(len(playlists), len(song_map)))

# Save file
save_npz(f_name_mx, sparseMatrix)

# print time taken
end = time.perf_counter()
print(f"\n\nTOTAL RUN DURATION: {end-start}")
