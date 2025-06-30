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


import src.utils as utils
import torch


class Node:
    """
    Implementation of a tree.
    """

    def __init__(self, low, high, num_factors, use_tensor_supp=False):
        """
        The value of a node is a subset of consecutive indices {low, ..., high - 1} included in
        {0, ..., num_factors - 1}.
        :param low: int
        :param high: int
        :param num_factors: int
        """
        self.low = low
        self.high = high
        self.num_factors = num_factors
        self.left = None  # Empty left child
        self.right = None  # Empty right child
        self.support = utils.partial_prod_butterfly_supports(
            num_factors, self.low, self.high
        )
        if use_tensor_supp:
            self.support = torch.from_numpy(self.support)

    def print_tree(self, level=0):
        """
        Method to print tree.
        :param level: int
        :return: str
        """
        ret = "\t" * level + str(self) + "\n"
        for child in [self.left, self.right]:
            if child:
                ret += child.print_tree(level + 1)
        return ret

    def to(self, *args, **kwargs):
        self.support = self.support.to(*args, **kwargs)
        if self.left is not None:
            self.left = self.left.to(*args, **kwargs)
        if self.right is not None:
            self.right = self.right.to(*args, **kwargs)
        return self

    def __str__(self):
        return f"[{self.high}; {self.low}]"

    def is_leaf(self):
        return self.left is None and self.right is None


def generate_balanced_tree(low, high, num_factors, use_tensor_supp=False):
    """
    Generate a balanced tree, with root's value {low, ..., high - 1}. num_factors corresponds to log_2(n), where
    n is the size of the matrix to factorize.
    :param low: int
    :param high: int
    :param num_factors: int
    :return: Node object
    """
    root = Node(low, high, num_factors, use_tensor_supp)
    if low < high - 1:
        split_index = (low + high) // 2
        root.left = generate_balanced_tree(
            split_index, high, num_factors, use_tensor_supp
        )
        root.right = generate_balanced_tree(
            low, split_index, num_factors, use_tensor_supp
        )
    return root


def generate_partial_balanced_tree(low, high, num_factors, depth, max_depth):
    root = Node(low, high, num_factors)
    if depth < max_depth and low < high - 1:
        split_index = (low + high) // 2
        root.left = generate_partial_balanced_tree(
            split_index, high, num_factors, depth + 1, max_depth
        )
        root.right = generate_partial_balanced_tree(
            low, split_index, num_factors, depth + 1, max_depth
        )
    return root


def generate_comb_tree(low, high, num_factors, use_tensor_supp=False):
    """
    Generate a comb tree, with root's value {low, ..., high - 1}. num_factors corresponds to log_2(n), where
    n is the size of the matrix to factorize.
    :param low: int
    :param high: int
    :param num_factors: int
    :return: Node object
    """
    root = Node(low, high, num_factors, use_tensor_supp)
    if low < high - 1:
        split_index = high - 1
        root.left = generate_comb_tree(split_index, high, num_factors, use_tensor_supp)
        root.right = generate_comb_tree(low, split_index, num_factors, use_tensor_supp)
    return root


def generate_partial_comb_tree(low, high, num_factors, depth, max_depth):
    root = Node(low, high, num_factors)
    if depth < max_depth and low < high - 1:
        split_index = high - 1
        root.left = generate_partial_comb_tree(
            split_index, high, num_factors, depth + 1, max_depth
        )
        root.right = generate_partial_comb_tree(
            low, split_index, num_factors, depth + 1, max_depth
        )
    return root


def generate_inversed_comb_tree(low, high, num_factors):
    root = Node(low, high, num_factors)
    if low < high - 1:
        split_index = low + 1
        root.left = generate_comb_tree(split_index, high, num_factors)
        root.right = generate_comb_tree(low, split_index, num_factors)
    return root


def generate_partial_inversed_comb_tree(low, high, num_factors, depth, max_depth):
    root = Node(low, high, num_factors)
    if depth < max_depth and low < high - 1:
        split_index = low + 1
        root.left = generate_partial_comb_tree(
            split_index, high, num_factors, depth + 1, max_depth
        )
        root.right = generate_partial_comb_tree(
            low, split_index, num_factors, depth + 1, max_depth
        )
    return root


def inorder(root):
    """
    Function to print the tree
    """
    if root:
        inorder(root.left)
        print(root, end=" ")
        inorder(root.right)


if __name__ == "__main__":
    num_factors = 5
    root = generate_balanced_tree(0, num_factors, num_factors)
    print(root.print_tree())
    print(root.left.support)
    print(root.left.left.support)
    print(root.left.right.support)
    print(root.right.support)
    inorder(root)

    root = generate_comb_tree(0, num_factors, num_factors)
    print(root.print_tree())
    print(root)
    print(root.left)
    print(root.right)
    print(root.right.right)
    print(root.right.left)
