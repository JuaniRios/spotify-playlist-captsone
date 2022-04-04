import time
from scipy.sparse import load_npz
from spotify_cleaning import *


if __name__ == "__main__":

    # %% Environment Variables
    # path = "../../ML_AI/ds_capstone/data/"  # Henrik
    path = "C:/Users/netzl/Offline Documents/spotify_million_playlist_dataset/data/"  # Daniel
    # path = "../data/spotify_million_playlist_dataset/data/" # Juan

    reduced_percentage = 2
    file_all_songs = "allSongs.pickle"  # name for song list pickle file

    file_MF_songs = f"songlist_for_MF_reduced_{reduced_percentage}.pickle"  # input songs for Matrix factorization
    file_leftout_songs = f"leftout_{reduced_percentage}.pickle"  # name for list of left out songs

    file_songmap_reduced = f'song_map_{reduced_percentage}.pickle' # reduced song_map
    file_mx_reduced = f'sparse_matrix_reduced_{reduced_percentage}.npz'  # name for MF input sparse matrix

    file_songmap = "song_map.pickle"  # full song_map
    file_mx_full = "sparseMatrix_full.npz"  # name for full matrix (used in EDA)


    start = time.perf_counter()

    # getting all songs (load or create)
    if not (os.path.isfile(file_all_songs)):
        print("Creating full playlist.")
        all_songs = dump_all_songs(file_all_songs, path)
    else:
        print("Full playlist found. Loading...")
        with open(file_all_songs, 'rb') as f:
            all_songs = pickle.load(f)

    # getting leftout and matrix factorization songs (load or create)
    if not (os.path.isfile(file_MF_songs) & os.path.isfile(file_leftout_songs)):
        print("Creating reduced playlists.")
        MF_songs, leftout_songs = create_playlists(file_MF_songs, file_leftout_songs, path, reduced_percentage = 2)
    else:
        print("Input and leftout playlist found. Loading...")
        with open(file_MF_songs, 'rb') as f:
            MF_songs = pickle.load(f)
        with open(file_leftout_songs, 'rb') as f:
            leftout_songs = pickle.load(f)

    # getting songmap (name => index)
    if not (os.path.isfile(file_songmap)):
        print("Creating full song map.")
        song_map = create_song_map(file_songmap, all_songs, percentage = 100)
    else:
        print("Full song map found. Loading...")
        with open(file_songmap, 'rb') as f:
            song_map = pickle.load(f)

    # getting the reduced version of songmap
    if not (os.path.isfile(file_songmap_reduced)):
        print("Creating reduced song map.")
        song_map_reduced = create_song_map(file_songmap_reduced, all_songs, percentage = reduced_percentage)
    else:
        print("Reduced song map found. Loading...")
        with open(file_songmap_reduced, 'rb') as f:
            song_map_reduced = pickle.load(f)

    # getting the full sparse matrix
    if not (os.path.isfile(file_mx_full)):
        print("Creating full sparse matrix.")
        sparseMatrix_full = create_sparse_matrix(file_mx_full, all_songs, song_map)
    else:
        print("Full sparse matrix found. Loading...")
        with open(file_mx_full, 'rb') as f:
            sparseMatrix_full = load_npz(f)

    # getting the reduced sparse matrix
    if not (os.path.isfile(file_mx_reduced)):
        print("Creating reduced sparse matrix.")
        sparseMatrix_reduced = create_sparse_matrix(file_mx_reduced, MF_songs, song_map_reduced)
    else:
        print("Reduced sparse matrix found. Loading...")
        with open(file_mx_reduced, 'rb') as f:
            sparseMatrix_reduced = load_npz(f)

    # print time taken
    end = time.perf_counter()
    print(f"\n\nTOTAL RUN DURATION: {end - start}")