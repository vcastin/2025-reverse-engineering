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
from pathlib import Path
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
import torch


def butterfly_support(size):
    """
    Generate a butterfly support of a given size
    :param size: int
    :return: numpy array
    """
    assert size % 2 == 0
    support = np.kron(np.ones((2, 2)), np.identity(size // 2))
    return support


def partial_prod_butterfly_supports(num_factors, low, high):
    """
    Closed form expression of partial matrix_product of butterfly supports. We name S_L, ..., S_1 the butterfly supports of
    size 2^L, represented as binary matrices. Then, the method computes the partial matrix_product S_{high-1} ... S_low.
    :param num_factors: int
    :param low: int
    :param high: int, excluded
    :return: numpy array, binary matrix
    """
    m = 2 ** (high - low)
    tmp = np.kron(np.ones((m, m)), np.identity(2**low))
    return np.kron(np.identity(2 ** (num_factors - high)), tmp)


def support_DFT(num_factors):
    """
    Generate the support for the butterfly factorization
    :param num_factors: int, number of factors in the butterfly factorization
    :return: list of numpy arrays, binary, each of size n x n, where n = 2 ** num_factors
    """
    return [
        partial_prod_butterfly_supports(num_factors, i - 1, i)
        for i in range(num_factors, 0, -1)
    ]


def perm_type(i, type):
    """
    Type 0 is c in paper. Type 1 is b in paper. Type 2 is a in paper.
    :param i:
    :param type:
    :return:
    """
    size = 2**i
    result = np.zeros((size, size))
    if type == 0:
        result[np.arange(size // 2), np.arange(size // 2)] = 1
        result[size // 2 + np.arange(size // 2), size - 1 - np.arange(size // 2)] = 1
    elif type == 1:
        result[size // 2 - 1 - np.arange(size // 2), np.arange(size // 2)] = 1
        result[size // 2 + np.arange(size // 2), size // 2 + np.arange(size // 2)] = 1
    else:
        result[np.arange(size // 2), np.arange(size // 2) * 2] = 1
        result[size // 2 + np.arange(size // 2), np.arange(size // 2) * 2 + 1] = 1
    return result


def shared_logits_permutation(num_factors, choices):
    """
    :param num_factors:
    :param choices: array of three bool
    :return:
    """
    permutations = []
    for i in range(2, num_factors + 1):
        block = np.identity(2**i)
        if choices[0]:
            block = block @ perm_type(i, 0)
        if choices[1]:
            block = block @ perm_type(i, 1)
        if choices[2]:
            block = block @ perm_type(i, 2)
        perm = np.kron(np.identity(2 ** (num_factors - i)), block)
        permutations.append(perm)
    return permutations


def perm_DFT(num_factors):
    result = []
    size = 2**num_factors
    for i in range(num_factors):
        if i == 0:
            continue
        for j in range(3):
            z = perm_type(i + 1, j)
            result.append(np.kron(np.identity(2 ** (num_factors - 1 - i)), z))
    return result


def generate_random_matrix(supp, main_m=1, main_std=0.5):
    """
    generate_matrix: generate matrix as described in the paper
    supp: list of factor supports
    main_m: mean of coefficients
    main_std: std of coefficients
    """
    matrix = np.identity(supp[0].shape[0])
    for support in supp:
        support = np.multiply(
            support, np.random.binomial(size=support.shape, n=1, p=0.5) * 2 - 1
        )
        matrix = matrix @ np.multiply(
            np.random.randn(support.shape[0], support.shape[1]) * main_std + main_m,
            support,
        )
    return matrix


def generate_matrix_noise(shape, noise_m, noise_std):
    """
    Generate random Gaussian matrix.
    :param shape: tuple (p, q), where p is the number of rows and q the number of columns.
    :param noise_m: mean of Gaussian additive noise to the factorized matrix
    :param noise_std: std of Gaussian additive noise to the factorized matrix
    :return: numpy array
    """
    noise = noise_m + np.random.randn(shape[0], shape[1]) * noise_std
    return noise


def product_of_factors(factors):
    B = np.identity(factors[0].shape[0])
    for f in factors:
        B = B @ f
    return B


def error_cal(factor, A, eps=1e-15, relative=True):
    """
    Calculate the error of approximation |A - S_1 S_2...S_n|
    :param factor: list of numpy arrays
    :param A: numpy array
    :param eps: float
    :param relative: bool
    :return: float
    """
    B = np.identity(A.shape[0])
    for f in factor:
        B = B @ f
    if relative:
        return np.linalg.norm(A - B) / (np.linalg.norm(A) + eps)
    return np.linalg.norm(A - B)


class Stats(object):
    """
    Log stuff with pandas library
    """

    def __init__(self, path, columns, rewrite=False):
        self.path = Path(path)
        self.stats = pd.DataFrame(columns=columns)

        if rewrite or not self.path.exists():
            self.stats = pd.DataFrame(columns=columns)
        else:
            # reload path stats
            self.stats = pd.read_pickle(self.path)
            # check that columns are the same
            assert list(self.stats.columns) == list(columns)

    def update(self, row, save=True):
        self.stats.loc[len(self.stats.index)] = row
        # save the statistics
        if save:
            self.stats.to_pickle(self.path)

    def update_with_dict(self, dictionary, save=True):
        self.stats = self.stats.append(dictionary, ignore_index=True)
        if save:
            self.stats.to_pickle(self.path)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def permute_columns(permutation, a):
    """
    Permutes the columns of the matrix a, given the permutation represented as an array
    :param permutation: array of int. For instance, [1, 0, 2, 3] (case n=4)
    :param a: matrix, numpy array
    :return: numpy array
    """
    # permutation = np.array(permutation, dtype=np.uint)
    idx = np.empty_like(permutation)
    idx[permutation] = np.arange(len(permutation))
    return a[:, idx]


def add_padding(A):
    n = A.shape[0]
    m = A.shape[1]
    p = 0
    while 2**p < m:
        p += 1
    m_pad = 2**p
    A_pad = np.zeros((n, m_pad))
    A_pad[:n, :m] = A
    return A_pad


def relative_error_padded_product(M, *factors, eps=1e-15):
    product_pad = matrix_product(factors)
    product = product_pad[: M.shape[0], : M.shape[1]]
    return np.linalg.norm(M - product) / (np.linalg.norm(M) + eps)


def matrix_product(factors):
    assert len(factors) > 0
    result = np.identity(factors[0].shape[0])
    for f in factors:
        result = result @ f
    return result


def best_perm(A, B):
    cost = np.square(distance.cdist(A.T, B.T, "euclidean"))
    row_ind, col_ind = linear_sum_assignment(cost)
    return col_ind


def inverse_bit(a, n):
    result = 0
    for i in range(n):
        result += ((a >> i) & 1) << (n - i - 1)
    return result


def bit_inverse_perm(n):
    result = []
    for i in range(2**n):
        result.append(inverse_bit(i, n))
    # print(result)
    return np.argsort(np.array(result))


def bit_reversal_permutation_matrix(num_factors):
    perm = np.zeros((2**num_factors, 2**num_factors))
    perm[np.arange(2**num_factors), bit_inverse_perm(num_factors)] = 1
    return perm


def get_permutation_matrix(num_factors, perm_name):
    """

    :param num_factors:
    :param perm_name: str, 000, 001, ..., 111
    :return:
    """
    if perm_name.isnumeric():
        choices = [int(char) for char in perm_name]
        p_list = shared_logits_permutation(num_factors, choices)
        p = product_of_factors(p_list)
    else:
        if perm_name == "identity":
            p = np.eye(2**num_factors)
        else:
            assert perm_name == "bit-reversal"
            p = bit_reversal_permutation_matrix(num_factors)
    return p


if __name__ == "__main__":
    print(get_permutation_matrix(3, "bit-reversal"))
    print(get_permutation_matrix(3, "001"))
