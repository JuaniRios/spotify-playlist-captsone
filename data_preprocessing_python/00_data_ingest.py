import json
import os
from scipy.sparse import vstack, save_npz, coo_matrix
import time
import pickle
from songlist_to_pickle import get_songlist


def make_inds(database):
    '''
    Creates lists of indices based on whether song exists on playlist).
    Returns parameters for creating a sparse matrix.
    :param database:
    :return:
    '''

    row_ix, col_ix = [], []
    for rowIdx, rw in enumerate(database):
        for dbSong in rw:
            row_ix.append(rowIdx)
            col_ix.append(song_map[dbSong])
    data_m = [1]*len(col_ix)
    return data_m, row_ix, col_ix


# SPECIFY HOW MANY FILES YOU WANT TO USE (n) AND FILENAMES FOR
# PICKLE SONGLIST (f_name_songlist) AND SPARSE MATRIX (f_name_mx)

start = time.time()
counter = 0
n = 1000
f_name_songlist = "allSongs_full.pickle"
f_name_mx = 'sparse_matrix_full.npz'


# create songlist
if not os.path.isfile(f_name_songlist):
    get_songlist(n, f_name_songlist)
    print("Created song pickle")
    with open(f_name_songlist, 'rb') as f:
        song_map = pickle.load(f)
    print("Pickle read")
else:
    with open(f_name_songlist, 'rb') as f:
        song_map = pickle.load(f)
    print("Pickle exists, read data from pickle")


print("parse json")
# parse json
path = "../../ML_AI/ds_capstone/data/"  # Henrik
# path = "C:/Users/netzl/Offline Documents/spotify_million_playlist_dataset/data/" # Daniel
db = []

for file in os.listdir(path):
    if counter == n:
        break

    db_start = time.time()
    data = json.load(open(path+file))
    for row in data["playlists"]:
        db.append([song["artist_name"] + " - " + song["track_name"] for song in row["tracks"]])  # 2D array?

    print("Read file nr.", counter)

    # create matrix

    data, row_inds, col_inds = make_inds(db)

    # sparsify matrix and either create new or append to existing sparseMatrix
    if counter == 0:
        sparseMatrix = coo_matrix((data, (row_inds, col_inds)), shape=(len(db), len(song_map)))

    else:
        temp_mx = coo_matrix((data, (row_inds, col_inds)), shape=(len(db), len(song_map)))
        sparseMatrix = vstack([sparseMatrix, temp_mx])
        del temp_mx

    # to save memory we flush db and create it again.
    del db
    db = []

    counter += 1
    db_end = time.time()
    print(f"One round time: {db_end - db_start}")

# Save file
save_npz(f_name_mx, sparseMatrix)

# print time taken
end = time.time()
print(f"\n\nTOTAL RUN DURATION: {end-start}")
