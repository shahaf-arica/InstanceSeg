import numpy as np
from scipy.linalg import eigh
import torch
from torch.functional import F


def ncut(A, tau=0.0, eps=1e-5, no_binary_graph=True, eig_vecs=8):
    if type(A) == torch.Tensor:
        A = A.cpu().numpy()
    if no_binary_graph:
        A[A < tau] = eps
    else:
        A = A > tau
        A = np.where(A.astype(float) == 0, eps, A)
    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)
    try:
        # Print second and third smallest eigenvector
        eigenvalues, eigenvectors = eigh(D - A, D, subset_by_index=[1, eig_vecs])
    except:
        # if eigh fails then D is not positive definite, and we should return None
        print("eigh failed")
        return None, None

    return eigenvectors, eigenvalues

def tokencut(features_1, features_2, tau=0.2, no_binary_graph=True, eig_vecs=2):
    features_1 = features_1[0, 1:, :]
    features_2 = features_2[0, 1:, :]
    features_1 = F.normalize(features_1, p=2)
    features_2 = F.normalize(features_2, p=2)
    A = (features_1 @ features_2.transpose(1, 0))

    eigenvectors, eigenvalues = ncut(A, tau=tau, no_binary_graph=no_binary_graph, eig_vecs=eig_vecs)

    return eigenvectors, eigenvalues


def tokencut_bipartition(features_1, features_2, tau=0.2, no_binary_graph=True):

    eigenvectors, eigenvalues = tokencut_bipartition(features_1, features_2, tau=tau, no_binary_graph=no_binary_graph)

    if eigenvectors is None or eigenvalues is None:
        return None, None
    else:

        # Using average point to compute bipartition
        second_smallest_vec = eigenvectors[:, 0]
        avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
        bipartition = second_smallest_vec > avg

        # seed = np.argmax(np.abs(second_smallest_vec))
        #
        # if bipartition[seed] != 1:
        #     second_smallest_vec = second_smallest_vec * -1
        #     bipartition = np.logical_not(bipartition)

        return bipartition, second_smallest_vec
