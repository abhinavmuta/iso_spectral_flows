"""Isospectral flow on Jacobi matrices.
"""
from argparse import ArgumentParser
import time

import numpy as np
from scipy.integrate import odeint

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from symm_mat import case, commute, plot, coords,


def rhs(X, t, N, dim):
    rhs = commute(X, commute(X, N, dim), dim)
    return np.ravel(rhs)


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--dim', action='store', type=int, dest='dim',
                   default=3, help='Size of the square matrix.')
    p.add_argument('--iters', action='store', type=float, dest='iters',
                   default=100, help='No of iterations.')
    p.add_argument('--plot', action='store', type=bool, dest='plot',
                   default=False, help='Plot the diagonal values of the matrix.')
    p.add_argument('--order', action='store', type=type, dest='order',
                   default=None,
                   help='Use ascending (0) or descending (1) order in N.')
    args = p.parse_args()


    dim = args.dim
    iters = args.iters
    times = np.linspace(0, iters, iters)
    A = np.random.rand(dim, dim)
    X0 = 0.5*(A + A.T)
    s = time.time()
    eigs_scipy = np.linalg.eigvals(X0)
    print(f"SciPy took: \t{time.time()-s:.2e}")
    eigs_scipy.sort()
    X0 = np.ravel(X0)
    N = case(dim, order=args.order)

    s = time.time()
    X = odeint(rhs, X0, times, args=(N, dim))
    print(f"Flow took: \t{time.time()-s:.2e}")
    Xf = X[-1].reshape(dim, dim)
    eigs_flow = np.diagonal(Xf).astype(float)
    order_of_eigs = np.argsort(np.argsort(eigs_flow))
    eigs_flow.sort()
    order_diff = np.diag(N) - order_of_eigs
    order_err = np.linalg.norm(order_diff)
    if order_err > 1e-2:
        print(np.diag(N)[abs(order_diff) > 0.1])
    print("Error in the order in which the eigenvalues appear to the order"
          " of values of N: ", order_err)
    print("Scipy: \t", eigs_scipy)
    print("Flow: \t", eigs_flow)
    eigs_err = np.linalg.norm(eigs_flow-eigs_scipy)
    print(f"Error in eigenvalues: \t{eigs_err:.2e}")
    matrix_err = np.linalg.norm(Xf - np.diag(np.diag(Xf)))
    print(f"Error in matrix: \t{matrix_err:.2e}")

    if args.dim < 3:
        x, y, z = coords(X, dim)
        plot(x, y, z, actual=eigs_scipy)
        plt.show()

    if args.plot:
        for i in range(dim):
            plt.plot(X[:, i*dim + i])
        plt.xlabel("Iterations")
        plt.show()
