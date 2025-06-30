# BSD 3-Clause License
#
# Copyright (c) 2022, ZHENG Leon
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np

import src.tree
from src import utils as utils


def retrieveCEC(support1, support2):
    """
    Partition the support to equivalent classes
    """
    assert support1.shape[1] == support2.shape[0]
    r = support1.shape[1]
    list_supp = []
    cec = []
    noncec = []

    for i in range(r):
        list_supp.append(list(support1[:, i]) + list(support2[i]))

    order = sorted(range(len(list_supp)), key=list_supp.__getitem__)
    index = 0
    while index < r:
        curr = [order[index]]
        i = index + 1
        while (
            i < r
            and (support1[:, order[i]] == support1[:, order[index]]).all()
            and (support2[order[i]] == support2[order[index]]).all()
        ):
            curr.append(order[i])
            i += 1
        if min(
            np.sum(support1[:, order[index]]), np.sum(support2[order[index]])
        ) <= len(curr):
            cec.append(curr)
        else:
            noncec.append(curr)
        index = i

    return cec, noncec


def best_low_rank(A, rank):
    """
    Finding the best low rank approximation by SVD
    """
    u, s, vh = np.linalg.svd(A)
    s = np.sqrt(s[:rank])
    return u[:, range(rank)] @ np.diag(s), np.diag(s) @ vh[range(rank)]


def solve_DTO(support1, support2, A, type="real"):
    """
    Algorithm 1
    :param support1: numpy array, binary matrix
    :param support2: numpy array, binary matrix
    :param A: numpy array
    :return: X, Y numpy arrays
    """
    cec, noncec = retrieveCEC(support1, support2)
    if type == "complex":
        X = np.zeros(support1.shape).astype(np.complex128)
        Y = np.zeros(support2.shape).astype(np.complex128)
    else:
        X = np.zeros(support1.shape)
        Y = np.zeros(support2.shape)
    for ce in cec:
        rep = ce[0]
        if np.sum(support1[:, rep]) == 0 or np.sum(support2[rep]) == 0:
            continue
        RP = np.where(support1[:, rep] == 1)[0]
        CP = np.where(support2[rep] == 1)[0]
        if len(ce) == len(RP) or len(ce) == len(RP):
            noncec.append(ce)
            continue
        submatrixA = A[RP][:, CP]
        if len(ce) >= len(RP):
            colx, rowx = np.meshgrid(ce, RP)
            coly, rowy = np.meshgrid(CP, ce[: len(RP)])
            X[rowx, colx] = np.eye(len(RP), len(ce))
            Y[rowy, coly] = submatrixA
        else:
            colx, rowx = np.meshgrid(ce[: len(CP)], RP)
            coly, rowy = np.meshgrid(CP, ce)
            X[rowx, colx] = submatrixA
            Y[rowy, coly] = np.eye(len(ce), len(CP))

    for ce in noncec:
        rep = ce[0]
        RP = np.where(support1[:, rep] == 1)[0]
        CP = np.where(support2[rep] == 1)[0]
        submatrixA = np.array(A[RP][:, CP])
        colx, rowx = np.meshgrid(ce, RP)
        coly, rowy = np.meshgrid(CP, ce)
        bestx, besty = best_low_rank(submatrixA, len(ce))
        X[rowx, colx] = bestx
        Y[rowy, coly] = besty
    return X, Y


def lifting_two_layers_factorization(support1, support2, A):
    """
    Lifting algorithm to factorize A into two factors with supports support1, support2, in the specific case
    where support1 and support2 have disjoint rank one supports.
    :param support1: numpy array, binary matrix
    :param support2: numpy array, binary matrix
    :param A: numpy array
    :return: X, Y are the left and right factors, as numpy arrays.
    """
    assert support1.shape[1] == support2.shape[0]
    dtype = np.complex128 if np.iscomplex(A).any() else np.float64
    X = np.zeros(support1.shape, dtype=dtype)
    Y = np.zeros(support2.shape, dtype=dtype)
    r = support1.shape[1]
    for t in range(r):
        rows = np.where(support1[:, t])[0]
        cols = np.where(support2[t, :])[0]
        subA = A[np.ix_(rows, cols)]
        u, v = best_low_rank(subA, 1)
        X[rows, t] = np.squeeze(u)
        Y[t, cols] = np.squeeze(v)
    return X, Y


def simple_hierarchical_factorization(support, A):
    """
    Hierarchical factorization approach in Section 5.2
    :param support: list of numpy arrays
    :param A: numpy array
    :return: list of numpy arrays
    """
    result = []
    matrix = A
    for i in range(len(support) - 1):
        support1 = support[i]
        support2 = np.identity(support[i].shape[1])
        for sp in support[i + 1 :]:
            support2 = support2 @ sp
            support2 = np.where(support2 > 0, 1, 0)
        X, Y = solve_DTO(support1, support2, matrix)
        result.append(X)
        matrix = Y
    result.append(matrix)
    return result


def tree_hierarchical_factorization(root, A, method="lifting"):
    """
    Method for hierarchical factorization described by a tree. We suppose that the sparsity constraints are the
    butterfly supports.
    :param root: Node object
    :param A: numpy array
    :param method: choice between 'lifting' or 'DTO'. Prefer 'lifting' since it is faster.
    :return: list of numpy arrays, representing the sparse factors of A.
    """
    assert not root.is_leaf()
    if method == "DTO":
        X, Y = solve_DTO(root.left.support, root.right.support, A)
    else:
        assert method == "lifting"
        X, Y = lifting_two_layers_factorization(
            root.left.support, root.right.support, A
        )
    left_factors = (
        [X] if root.left.is_leaf() else tree_hierarchical_factorization(root.left, X)
    )
    right_factors = (
        [Y] if root.right.is_leaf() else tree_hierarchical_factorization(root.right, Y)
    )
    return left_factors + right_factors


def project_BP_model_P_fixed(
    matrix, tree_type, p=None, max_depth=-1, return_factors=False, return_root=False
):
    generate_partial_tree, generate_tree = get_generation_tree_methods(tree_type)
    num_factors = int(np.log2(matrix.shape[1]))
    if max_depth >= 0:
        root = generate_partial_tree(0, num_factors, num_factors, 0, max_depth)
    else:
        root = generate_tree(0, num_factors, num_factors)
    if p is not None:
        factors = tree_hierarchical_factorization(root, matrix @ np.transpose(p))
        product = utils.product_of_factors(factors) @ p
    else:
        factors = tree_hierarchical_factorization(root, matrix)
        product = utils.product_of_factors(factors)
    if return_factors:
        if return_root:
            return product, factors, root
        return product, factors
    if return_root:
        return product, root
    return product


def get_generation_tree_methods(tree_type):
    if tree_type == "comb":
        generate_partial_tree = src.tree.generate_partial_comb_tree
        generate_tree = src.tree.generate_comb_tree
    elif tree_type == "inversed_comb":
        generate_partial_tree = src.tree.generate_partial_inversed_comb_tree
        generate_tree = src.tree.generate_inversed_comb_tree
    else:
        assert tree_type == "balanced"
        generate_partial_tree = src.tree.generate_partial_balanced_tree
        generate_tree = src.tree.generate_balanced_tree
    return generate_partial_tree, generate_tree


def project_BP_model_8_perm_fixed(
    matrix, tree_type, max_depth=-1, return_factors=False, return_root=False
):
    num_factors = int(np.log2(matrix.shape[1]))
    permutations = [
        utils.get_permutation_matrix(num_factors, perm_name)
        for perm_name in ["000", "001", "010", "011", "100", "101", "110", "111"]
    ]
    # print(permutations)
    projections = [
        project_BP_model_P_fixed(
            matrix, tree_type, p, max_depth, return_factors, return_root
        )
        for p in permutations
    ]
    if return_factors or return_root:
        errors = [
            np.linalg.norm(matrix - projection[0]) / np.linalg.norm(matrix)
            for projection in projections
        ]
    else:
        errors = [
            np.linalg.norm(matrix - projection) / np.linalg.norm(matrix)
            for projection in projections
        ]
    print(errors)
    argmin_error = np.argmin(errors)
    return (*projections[argmin_error], permutations[argmin_error])


"""
if __name__ == '__main__':
    import scipy
    n = 9
    matrix = scipy.linalg.hadamard(2 ** n)# @ utils.bit_reversal_permutation_matrix(n).T
    support = utils.support_DFT(n)
    result = simple_hierarchical_factorization(support, matrix)
    print(utils.error_cal(result, matrix))
"""
