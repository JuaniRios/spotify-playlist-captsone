import json
import numpy as np
from scipy.sparse import csr_matrix
from sys import getsizeof
from itertools import chain
import time

start = time.time()
print("read data")
# read data
data = json.load(open('../../ML_AI/ds_capstone/data/mpd.slice.0-999.json'))

print("parse")
# parse json
db = []
for row in data["playlists"]:
    # p_id = row["pid"]
    tracks = [song["artist_name"] + " - " + song["track_name"] for song in row["tracks"]]
    db.append(tracks)

print("get unique songs")
# get unique songs
song_list = list(set(chain(*db)))

song_map = {}
for i, song in enumerate(song_list):
    song_map[song] = i

print("Turn into matrix")

# pre-fill matrix with zeroes

mx = np.zeros((len(db), len(song_list)), dtype="int").tolist()


for rowIdx, row in enumerate(db):
    for dbSong in row:
        mx[rowIdx][song_map[dbSong]] = 1

print("checksum db:", len(set(db[55])))
print("checksum mx:", sum(mx[55]))

# to sparse matrix
# sparseMatrix = csr_matrix(mx)

# Compare usage of variables
# print(getsizeof(mx))

# print(getsizeof(sparseMatrix))

# print("mx[0]", mx[0])

end = time.time()
print(f"Total time: {end-start}")
