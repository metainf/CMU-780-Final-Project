from time import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets

# ============== Loading my own files =================
loaded = np.load("5_square_board_data.npz")

vis_array = loaded['vis_array']
vis_counts_array = loaded['vis_counts_array']

boards_array = loaded['boards_array']
boards_counts_array = loaded['boards_counts_array']

X = vis_array
Y = vis_counts_array
#  X = boards_array
#  Y = boards_counts_array
indices = np.random.choice(len(X), 5000, replace=False)
X = X[indices]
Y = Y[indices]

# Determine all of the different values counts
different_counts = sorted(list(set(Y.reshape(-1).tolist())))
count_mapping = {different_counts[i]:i for i in range(len(different_counts))}
count_array = [count_mapping[count] for count in Y.reshape(-1)]
count_array = np.array(count_array, dtype=float)
print(count_mapping)

# Matching the variables below
color = count_array

# =============== Visualization code borrowed from the internets =======
# Author: Jake Vanderplas -- <vanderplas@astro.washington.edu>
n_neighbors = 10
n_components = 2

fig = plt.figure(figsize=(15, 8))
plt.suptitle("Manifold Learning with %i points, %i neighbors"
             % (1000, n_neighbors), fontsize=14)


ax = fig.add_subplot(251, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.view_init(4, -72)

methods = ['standard', 'ltsa', 'hessian', 'modified']
labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']

for i, method in enumerate(methods):
    t0 = time()
    try:
        Y = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                            eigen_solver='auto',
                                            method=method).fit_transform(X)
    except:
        continue

    t1 = time()
    print("%s: %.2g sec" % (methods[i], t1 - t0))

    ax = fig.add_subplot(252 + i)
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.title("%s (%.2g sec)" % (labels[i], t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

t0 = time()
Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
t1 = time()
print("Isomap: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(257)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("Isomap (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


t0 = time()
mds = manifold.MDS(n_components, max_iter=100, n_init=1)
Y = mds.fit_transform(X)
t1 = time()
print("MDS: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(258)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("MDS (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


t0 = time()
se = manifold.SpectralEmbedding(n_components=n_components,
                                n_neighbors=n_neighbors)
Y = se.fit_transform(X)
t1 = time()
print("SpectralEmbedding: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(259)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("SpectralEmbedding (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

#  t0 = time()
#  tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
#  Y = tsne.fit_transform(X)
#  t1 = time()
#  print("t-SNE: %.2g sec" % (t1 - t0))
#  ax = fig.add_subplot(2, 5, 10)
#  plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
#  plt.title("t-SNE (%.2g sec)" % (t1 - t0))
#  ax.xaxis.set_major_formatter(NullFormatter())
#  ax.yaxis.set_major_formatter(NullFormatter())
#  plt.axis('tight')

plt.show()
