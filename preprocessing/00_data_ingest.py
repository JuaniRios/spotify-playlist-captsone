import time
import os
from decouple import config

os.chdir(config("PROJECT_PATH"))
from preprocessing.cleaning_functions import *

# %% Environment Variables
dataset_path = config("PROJECT_PATH") + "/data/spotify_million_playlist_dataset/data/"

reduced_percentage = int(config("REDUCED_PERCENTAGE"))
min_song_count = int(config("MIN_SONG_COUNT"))
cutoff = float(config("CUTOFF"))
min_songs = int(config("MIN_SONGS"))

file_all_songs = config("FILE_ALL_SONGS")
file_songmap = config("FILE_SONGMAP")
file_mx_full = config("FILE_MX_FULL")

file_MF_songs = config("FILE_MF_SONGS").format(reduced_percentage=reduced_percentage)
file_leftout_songs = config("FILE_LEFTOUT_SONGS").format(reduced_percentage=reduced_percentage)

file_songmap_reduced = config("FILE_SONGMAP_REDUCED").format(reduced_percentage=reduced_percentage)
file_mx_reduced = config("FILE_MX_REDUCED").format(reduced_percentage=reduced_percentage)

# %% Check for already created files, if they are not present, create them
start = time.perf_counter()

# getting all songs (load or create)
if not (os.path.isfile(file_all_songs)):
    print("Creating full playlist.")
    all_songs = dump_all_songs(file_all_songs, dataset_path)
else:
    print("Full playlist found.")
    with open(file_all_songs, 'rb') as f:
        all_songs = pickle.load(f)

# getting songmap (name => index)
if not (os.path.isfile(file_songmap)):
    print("Creating full song map.")
    song_map = create_song_map(file_songmap, all_songs, percentage=100)
else:
    print("Full song map found.")
    with open(file_songmap, 'rb') as f:
        song_map = pickle.load(f)

# getting the reduced version of songmap
if not (os.path.isfile(file_songmap_reduced)):
    print("Creating reduced song map.")
    song_map_reduced = create_song_map(file_songmap_reduced, all_songs, percentage=reduced_percentage,
                                       min_song_count=min_song_count)
else:
    print("Reduced song map found.")
    with open(file_songmap_reduced, 'rb') as f:
        song_map_reduced = pickle.load(f)

# getting leftout and matrix factorization songs (load or create)
if not (os.path.isfile(file_MF_songs) & os.path.isfile(file_leftout_songs)):
    print("Creating reduced playlists.")
    MF_songs, leftout_songs = create_playlists(file_MF_songs, file_leftout_songs, dataset_path, song_map_reduced,
                                               reduced_percentage=reduced_percentage, cutoff_factor=cutoff,
                                               min_songs=min_songs)
else:
    print("Input and leftout playlist found.")
    with open(file_MF_songs, 'rb') as f:
        MF_songs = pickle.load(f)
    with open(file_leftout_songs, 'rb') as f:
        leftout_songs = pickle.load(f)

# getting the full sparse matrix
if not (os.path.isfile(file_mx_full)):
    print("Creating full sparse matrix.")
    sparseMatrix_full = create_sparse_matrix(file_mx_full, all_songs, song_map)
else:
    print("Full sparse matrix found.")

# getting the reduced sparse matrix
if not (os.path.isfile(file_mx_reduced)):
    print("Creating reduced sparse matrix.")
    sparseMatrix_reduced = create_sparse_matrix(file_mx_reduced, MF_songs, song_map_reduced)
else:
    print("Reduced sparse matrix found.")

# print time taken
end = time.perf_counter()
print(f"\n\nTOTAL RUN DURATION: {end - start}")
