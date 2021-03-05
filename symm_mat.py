"""
Isospectral flow on symmetric matrices.
"""
from argparse import ArgumentParser
import time

import numpy as np
from scipy.integrate import solve_ivp, odeint

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def commute(A, B, dim):
    A = A.reshape(dim, dim)
    B = B.reshape(dim, dim)
    return B@A - A@B


def rhs(t, X, N, dim):
    rhs = commute(X, commute(X, N, dim), dim)
    return np.ravel(rhs)


def case(dim, order=None):
    vals = np.linspace(0, dim-1, dim)
    if order is not None:
        reverse = False if order is 0 else True
        N = np.diag(sorted(vals, reverse=reverse))
    else:
        np.random.shuffle(vals)
    N = np.diag(vals)
    return N


def plot(x, y, z, actual):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x[0], y[0], z[0], marker='*', color='r')
    ax.text(x[0], y[0], z[0], 'Start', size=10, zorder=1, color='r')
    ax.scatter(x[-1], y[-1], z[-1], marker='+', color='r')
    ax.text(x[-1], y[-1], z[-1]-0.01, 'End', size=10, zorder=1, color='r')
    ax.scatter(actual[0], actual[1], 0, marker='8', color='g')
    ax.text(actual[0], actual[1], 0.01, 'Actual', size=10, zorder=1,
            color='g')
    ax.plot(x, y, z)
    ax.set_xlabel('A11')
    ax.set_ylabel('A12')
    ax.set_zlabel('A22')


def coords(X_all_times, dim):
    x, y, z = np.zeros((3, X_all_times.shape[0]))
    for i in range(X_all_times.shape[0]):
        m = X_all_times[i].reshape(dim, dim)
        d = np.diagonal(m).copy()
        d.sort()
        x[i], y[i] = d
        z[i] = X_all_times[i][1]
    return x, y, z


def initialize(dim):
    A = np.random.rand(dim, dim)
    return 0.5*(A + A.T)


def solve_scipy(X0):
    s = time.time()
    eigs_scipy = np.linalg.eigvals(X0)
    print(f"SciPy took: \t{time.time()-s:.2e}s")
    eigs_scipy.sort()
    return eigs_scipy


def solve_flow_ivp(X0, order):
    X0 = X0.ravel()
    N = case(dim, order=order)

    s = time.time()
    X = solve_ivp(
        rhs, times, X0, args=(N, dim,), method='LSODA', rtol=1e-2, atol=1e-2
    )
    print(f"IVP took: \t{time.time()-s:.2e}s")
    return X.y.T, N


def solve_flow(X0, order):
    X0 = X0.ravel()
    N = case(dim, order=order)

    s = time.time()
    X = odeint(rhs, X0, times, args=(N, dim,), tfirst=True)
    print(f"Flow took: \t{time.time()-s:.2e}s")
    return X, N


def get_eigenvalues(X, N):
    X_all_times = X[-1].reshape(dim, dim)
    eigs_flow = np.diagonal(X_all_times).astype(float)

    order_of_eigs = np.argsort(np.argsort(eigs_flow))
    eigs_flow.sort()

    order_diff = np.diag(N) - order_of_eigs
    order_error = np.linalg.norm(order_diff)
    if order_error > 1e-2:
        print(np.diag(N)[abs(order_diff) > 0.1])

    return eigs_flow, order_of_eigs, order_error


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--dim', action='store', type=int, dest='dim',
                   default=3, help='Size of the square matrix.')
    p.add_argument('--iters', action='store', type=float, dest='iters',
                   default=100, help='No of iterations.')
    p.add_argument('--plot', action='store_true', dest='plot',
                   default=False,
                   help='Plot the diagonal values of the matrix.')
    p.add_argument('--order', action='store', type=type, dest='order',
                   default=None,
                   help='Use ascending (0) or descending (1) order in N, if \
                   not specified a random shuffle is made.')
    args = p.parse_args()

    dim = args.dim
    iters = args.iters
    times = np.linspace(0, 1, iters)

    X0 = initialize(dim)
    X0 = np.array([[0.30466469, 0.35395261, 0.48226515, 0.49330274, 0.46455406],
                   [0.35395261, 0.72538645, 0.48997522, 0.81385778, 0.42610372],
                   [0.48226515, 0.48997522, 0.54678563, 0.748202, 0.58075574],
                   [0.49330274, 0.81385778, 0.748202, 0.97722199, 0.58034516],
                   [0.46455406, 0.42610372, 0.58075574, 0.58034516, 0.75478618]])

    X0 = np.array([[4, 0, 0, 0, 0],
                   [0, 72, 0, 0, 0],
                   [0, 0, 54, 0, 0],
                   [0, 0, 0, 91, 0],
                   [0, 0, 0, 0, 70]])


    X_all_times, N = solve_flow(X0, order=args.order)

    X_all_times_ivp, N_ivp = solve_flow_ivp(X0, order=args.order)

    eigs_flow, order_of_eigs, order_error = get_eigenvalues(X_all_times, N)
    eigs_flow_ivp, _ = get_eigenvalues(X_all_times_ivp, N_ivp)

    eigs_scipy = solve_scipy(X0)

    print("Flow: \t", eigs_flow)
    print("IVP: \t", eigs_flow_ivp)
    print("SciPy: \t", eigs_scipy)

    print("Error in the order in which the eigenvalues appear vs. the order"
          " of values of N: ", order_error)

    eigs_err = np.linalg.norm(eigs_flow-eigs_scipy)
    print(f"Difference in eigenvalues of SciPy and Flow: \t{eigs_err:.2e}")

    if args.dim < 3:
        x, y, z = coords(X_all_times, dim)
        plot(x, y, z, actual=eigs_scipy)
        plt.show()

    if args.plot:
        for i in range(dim):
            plt.plot(X_all_times[:, i*dim + i])
        plt.title("The values of diagonal #'s vs. no of Iterations.")
        plt.xlabel("Iterations")
        plt.show()

        plt.plot(eigs_scipy, '+r', label='SciPy')
        plt.plot(eigs_flow, 'xk', label='Flow')
        plt.title("Eigenvalues of SciPy vs. the flow.")
        plt.legend()
        plt.show()
