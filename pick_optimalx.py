import numpy as np

file_name = 'train_data.txt'
X = np.loadtxt(file_name)
min_idx = np.argmin(X[:, -1])

k_opt = X[min_idx, 0]
R_opt = X[min_idx, 1]

print k_opt
print R_opt