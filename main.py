import os
import numpy as np
from time import time

if __name__ == "__main__":

    # Let's take the randomness out of random numbers (for reproducibility)
    np.random.seed(0)

    # for numpy
    size = 4096
    A, B = np.random.random((size, size)), np.random.random((size, size))
    C, D = np.random.random((size * 128,)), np.random.random((size * 128,))
    E = np.random.random((int(size / 2), int(size / 4)))
    F = np.random.random((int(size / 2), int(size / 2)))
    F = np.dot(F, F.T)
    G = np.random.random((int(size / 2), int(size / 2)))

    # for data strage access
    n_ch = 64
    n_t = 1000*40*60 # 30 minutes

    # Matrix multiplication
    N = 20
    t = time()
    for i in range(N):
        np.dot(A, B)
    delta = time() - t
    print('Dotted two %dx%d matrices in %0.2f s.' % (size, size, delta / N))
    del A, B

    # Vector multiplication
    N = 5000
    t = time()
    for i in range(N):
        np.dot(C, D)
    delta = time() - t
    print('Dotted two vectors of length %d in %0.2f ms.' % (size * 128, 1e3 * delta / N))
    del C, D

    # Singular Value Decomposition (SVD)
    N = 3
    t = time()
    for i in range(N):
        np.linalg.svd(E, full_matrices = False)
    delta = time() - t
    print("SVD of a %dx%d matrix in %0.2f s." % (size / 2, size / 4, delta / N))
    del E

    # Cholesky Decomposition
    N = 3
    t = time()
    for i in range(N):
        np.linalg.cholesky(F)
    delta = time() - t
    print("Cholesky decomposition of a %dx%d matrix in %0.2f s." % (size / 2, size / 2, delta / N))

    # Eigendecomposition
    t = time()
    for i in range(N):
        np.linalg.eig(G)
    delta = time() - t
    print("Eigendecomposition of a %dx%d matrix in %0.2f s." % (size / 2, size / 2, delta / N))

    # write large file
    N = 10
    data = np.random.random((n_ch, n_t))
    data = data.astype(np.float64)
    t = time()
    for i in range(N):
        np.save(os.path.join(os.path.expanduser('~'), "benchmark.npy"), data)
    delta = time() - t
    print("write a large file of a %dx%d matrix in %0.2f s." % (n_ch, n_t, delta/N))

    # read large file
    N = 10
    t = time()
    for i in range(N):
        data = np.load(os.path.join(os.path.expanduser('~'), "benchmark.npy"))
    delta = time() - t
    print("read a large file of a %dx%d matrix in %0.2f s." % (n_ch, n_t, delta/N))

    # write small file
    N = 1000
    data = np.random.random((n_ch, n_ch))
    data = data.astype(np.float64)
    t = time()
    for i in range(N):
        np.save(os.path.join(os.path.expanduser('~'), "benchmark.npy"), data)
    delta = time() - t
    print("write a small file of a %dx%d matrix in %0.5f ms." % (n_ch, n_ch, (delta/N)*1000))

    # read small file
    N = 1000
    t = time()
    for i in range(N):
        data = np.load(os.path.join(os.path.expanduser('~'), "benchmark.npy"))
    delta = time() - t
    print("read a small file of a %dx%d matrix in %0.5f ms." % (n_ch, n_ch, (delta/N)*1000))

    if os.path.exists(os.path.join(os.path.expanduser('~'), "benchmark.npy")):
        os.remove(os.path.join(os.path.expanduser('~'), "benchmark.npy"))