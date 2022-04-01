import json
import os
from itertools import chain
from scipy.sparse import save_npz, coo_matrix
import time
import pickle
from ordered_set import OrderedSet


#%% Environment Variables
# path = "../../ML_AI/ds_capstone/data/"  # Henrik
path = "C:/Users/netzl/Offline Documents/spotify_million_playlist_dataset/data/" # Daniel
# path = "../data/spotify_million_playlist_dataset/data/" # Juan
reduced_percentage = 2
f_name_songlist = "allSongs_reduced_2.pickle" # name for song list pickle file
f_name_songslist_for_MF = "songlist_for_MF_reduced_2.pickle" # allSongs - leftout -> for MF
f_name_mx = 'sparse_matrix_reduced_2.npz' # name for sparse matrix
f_name_song_map = "song_map_2.pickle" # name for song_map pickle (to keep the order of songs in mx)
f_name_leftout = "leftout_2.pickle" # name for list of left out songs
start = time.perf_counter()
#os.chdir("/data_preprocessing_python")

#%% Create list of playlists with all songs, with songs for MF only and with leftout songs
def pickle_playlists(filename_all, filename_MF_only, f_name_leftout, data_loc, percentage):
    print(os.getcwd())
    db = [] # all songs
    MF_songs = [] # songs for MF only
    leftout_songs = [] # track all songs which are left out of each playlist; appending lists
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
            # define percentage of songs which will be used for MF in each playlist
            leftout_cutoff = int(round(len(row) * .9, 0))  # 90% of each playlist for creating recommendation
            # create tracks list
            tracks = [song["artist_name"] + " - " + song["track_name"] for song in row["tracks"]]
            db.append(tracks) # store all songs
            # append 90% to tracks list used for MF
            MF_songs.append(tracks[:leftout_cutoff]) # first 90%
            # append 10% of songs in each playlist to leftout_songs for evaluation
            leftout_songs.append(tracks[leftout_cutoff:]) # last 10% for evaluation/calculating hit rate

    with open(filename_all, 'wb') as f:
        # save all tracks
        pickle.dump(db, f, pickle.HIGHEST_PROTOCOL)

    with open(filename_MF_only, 'wb') as f:
        # save tracks for MF
        pickle.dump(MF_songs, f, pickle.HIGHEST_PROTOCOL)

    with open(f_name_leftout, 'wb') as f:
        # save left out tracks for evaluation
        pickle.dump(leftout_songs, f, pickle.HIGHEST_PROTOCOL)

# check if full playlist, playlist for MF and left out songs for evaluation are present; if not, create them
if not (os.path.isfile(f_name_songlist) & os.path.isfile(f_name_leftout) & os.path.isfile(f_name_songslist_for_MF)):
    print("Pickle not found, creating it now. Please wait...")
    pickle_playlists(f_name_songlist, f_name_songslist_for_MF, f_name_leftout, path, reduced_percentage)
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

song_list = list(OrderedSet(chain(*playlists))) # Create list of songs without duplicates; preserve order for evaluation
print(f"Length of song_list: {len(song_list)}")
#print(song_list[:5])

# create song_map to map each song to its index
song_map = {song:i for i,song in enumerate(song_list)}

# Save Songmap
with open(f_name_song_map, 'wb') as f:
     pickle.dump(song_map, f, pickle.HIGHEST_PROTOCOL)

# load songs for MF only to create matrix
with open(f_name_songslist_for_MF, 'rb') as f:
    playlists_for_MF = pickle.load(f)

# create vectors for sparse matrix
# data contains only 1s, row_ix contains playlist indices, col_ix contains song indices
data_m, row_ix, col_ix = [], [], []

for rowIdx, rw in enumerate(playlists_for_MF):
    # for each playlist
    for dbSong in rw:
        # for each song in a playlist
        row_ix.append(rowIdx) # row index
        col_ix.append(song_map[dbSong]) # song index
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
