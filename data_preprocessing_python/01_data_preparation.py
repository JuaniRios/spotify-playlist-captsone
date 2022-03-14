#%% Imports
import os

import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
os.chdir("C:/Users/juani/PycharmProjects/spotify-playlist-captsone/data_preprocessing_python")

#%% Load Matrix
print(os.getcwd())
print("loading matrix from file")
matrix = sparse.load_npz("./sparse_matrix_reduced.npz")
print("finished loading matrix")

#%% plot matrix
plt.spy(matrix)
plt.show()

#%% test
A = sparse.rand(10000,10000, density=0.00001)
M = sparse.csr_matrix(A)
plt.spy(M)
plt.show()
