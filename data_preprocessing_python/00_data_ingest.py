import json
import os
from scipy.sparse import csr_matrix, vstack, save_npz
from sys import getsizeof
import time
import pickle
from songlist_to_pickle import get_songlist


def make_mx(database):
    '''
    Fill mx with zeroes, and fill in 1's to correct indexes
    returns matrix
    :param database:
    :return:
    '''
    # pre-fill matrix with zeroes

    mx = []
    # pre-fill with zeroes for width of length of songs
    for i in range(len(database)):
        mx.append([0] * len(song_map))

    # fill 1's to indexes
    for rowIdx, rw in enumerate(database):
        for dbSong in rw:
            mx[rowIdx][song_map[dbSong]] = 1

    return mx


# SPECIFY HOW MANY FILES YOU WANT TO USE (n) AND FILENAMES FOR
# PICKLE SONGLIST (f_name_songlist) AND SPARSE MATRIX (f_name_mx)

start = time.time()
counter = 0
n = 10
f_name_songlist = 'data10k.pickle'
f_name_mx = 'sparse_matrix10k.npz'


# create songlist
get_songlist(n, f_name_songlist)

print("read songs")
# get unique songs
with open(f_name_songlist, 'rb') as f:
    song_map = pickle.load(f)


print("parse json")
# parse json
db = []
for file in os.listdir("../../ML_AI/ds_capstone/data/"):
    if counter == n:
        break

    db_start = time.time()
    data = json.load(open('../../ML_AI/ds_capstone/data/'+file))
    for row in data["playlists"]:
        tracks = [song["artist_name"] + " - " + song["track_name"] for song in row["tracks"]]
        db.append(tracks)

    print("Read file nr.", counter)

    # create matrix
    temp_mx = make_mx(db)

    # sparsify matrix and either create new or append to existing sparseMatrix
    if counter == 0:
        sparseMatrix = csr_matrix(temp_mx)
    else:
        sparseMatrix = vstack([sparseMatrix, temp_mx])
        print(getsizeof(sparseMatrix))

    # to save memory we flush db and create it again.
    # TODO: find better solution for saving memory. Garbage collection?
    del db
    db = []

    counter += 1
    db_end = time.time()
    print(f"One round time: {db_end - db_start}")

# Save file
save_npz(f_name_mx, sparseMatrix)


end = time.time()
print(f"\n\nTOTAL RUN DURATION: {end-start}")
