import json
import os
from itertools import chain

from scipy.sparse import vstack, save_npz, coo_matrix
import time
import pickle

#%% Environment Variables
# path = "../../ML_AI/ds_capstone/data/"  # Henrik
# path = "C:/Users/netzl/Offline Documents/spotify_million_playlist_dataset/data/" # Daniel
path = "../data/spotify_million_playlist_dataset/data/" # Juan

f_name_songlist = "allSongs_full.pickle" # name for song list pickle file
f_name_mx = 'sparse_matrix_full.npz' # name for sparse matrix
start = time.perf_counter()

#%% Create list of playlists
def pickle_playlists(filename, data_loc):
    db = []
    # get all files (dataset is divided)
    for file in os.listdir(data_loc):
        print(f"getting {file}")
        data = json.load(open(data_loc + file))
        # list of playlist are under the key "playlists"
        for row in data["playlists"]:
            tracks = [song["artist_name"] + " - " + song["track_name"] for song in row["tracks"]]
            db.append(tracks)

    with open(filename, 'wb') as f:
        pickle.dump(db, f, pickle.HIGHEST_PROTOCOL)

if not os.path.isfile(f_name_songlist):
    print("Pickle not found, creating it now. Please wait...")
    pickle_playlists(f_name_songlist, path)
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
song_list = list(set(chain(*playlists)))
song_map = {song:i for i,song in enumerate(song_list)}

data_m, row_ix, col_ix = [], [], []

for rowIdx, rw in enumerate(playlists):
    for dbSong in rw:
        row_ix.append(rowIdx)
        col_ix.append(song_map[dbSong])
data_m = [1]*len(col_ix)

#%% Create Sparse Matrix
print("Creating Sparse Matrix")
sparseMatrix = coo_matrix((data_m, (row_ix, col_ix)), shape=(len(playlists), len(song_map)))

# Save file
save_npz(f_name_mx, sparseMatrix)

# print time taken
end = time.perf_counter()
print(f"\n\nTOTAL RUN DURATION: {end-start}")
