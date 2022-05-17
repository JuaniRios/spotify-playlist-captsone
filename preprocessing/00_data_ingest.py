import time
import os
from decouple import config

os.chdir(config("PROJECT_PATH"))
from preprocessing.cleaning_functions import *

# %% Environment Variables
dataset_path = config("PROJECT_PATH") + "/data/spotify_million_playlist_dataset/data/"

reduced_percentage = 2  # reduce dataset by 100-reduced_percentage %. so this var is the actual size compared to original
min_song_count = 1  # include in dataset only songs (columns) that appear x amount of times in the dataset
cutoff = 0.9  # train/test data ratio. train = x * dataset, test = 1-x * dataset
min_songs = 1  # include in dataset only playlists (rows) that contain x amount of songs

file_all_songs = "allSongs.pickle"  # file name for song list pickle file
file_songmap = "song_map.pickle"  # file name of dict of song_name: col_index in matrix.
file_mx_full = "sparseMatrix_full.npz"  # file name for full matrix (used in EDA)

file_MF_songs = f"songlist_for_MF_reduced_{reduced_percentage}.pickle"  # file name of input songs for Matrix factorization
file_leftout_songs = f"leftout_{reduced_percentage}.pickle"  # file name of list of left out songs (used for testing)

file_songmap_reduced = f'song_map_{reduced_percentage}.pickle'  # filename for reduced songmap for MF
file_mx_reduced = f'sparse_matrix_reduced_{reduced_percentage}.npz'  # file name of MF input sparse matrix

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
