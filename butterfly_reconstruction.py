# %%
import numpy as np
import matplotlib.pyplot as plt
from src.utils import partial_prod_butterfly_supports
from toolbox import ReLU, factorization_with_missing_values, support_matrix, create_grid

# %% Defining the butterfly network

np.random.seed(25)

L = 3
d = 2**L

W = [np.random.randn(d, d) for _ in range(L)]
for j in range(L):
    S_j = partial_prod_butterfly_supports(L, j, j + 1)
    W[j] = W[j] * S_j


def forward(x, W=W, store_activations=True):
    # W is a list of butterfly matrices
    activations = []
    for W_layer in W[:-1]:
        x = W_layer @ x
        x = ReLU(x)
        if store_activations:
            activations.append(np.where(x > 0, 1, 0))
    return W[-1] @ x, activations


def jacobian(activations, W=W):
    J = np.eye(d)
    for i, W_layer in enumerate(W[:-1]):
        J = W_layer @ J
        J = np.diag(activations[i]) @ J
    J = W[-1] @ J
    return J


def product_of_weights(W):
    product = np.eye(d)
    for W_layer in W:
        product = W_layer @ product
    return product


s = np.zeros(L * d**2)
for j in range(L):
    s_j = partial_prod_butterfly_supports(L, j, j + 1).flatten()
    s[j * d**2 : (j + 1) * d**2] = s_j

non_zero_indices = np.where(s != 0)[0]  # vectorized support


def P_operator(log_w):
    # w is vect(W_1),...,vect(W_J) stacked (without zero coordinates)
    # log_w is np.real(np.emath.log(w))
    w = np.exp(log_w)
    w_full = np.zeros(L * d**2)
    w_full[non_zero_indices] = w
    W = w_full.reshape((L, d, d))
    W_product = product_of_weights(W)
    out = np.emath.log(W_product.flatten()).real
    return out


def P_matrix(P_operator):
    P = np.zeros((d**2, 2 * d * L))
    for i in range(2 * L * d):
        e_i = np.zeros(2 * L * d)
        e_i[i] = 1
        P[:, i] = P_operator(e_i)
    return P


P = P_matrix(P_operator)


def paths_containing_neuron(layer, index):
    e_il = np.zeros(2 * d * L)
    e_il[2 * layer * d + 2 * index] = 1
    e_il[2 * layer * d + 2 * index + 1] = 1
    flattened_indices = np.where(P @ e_il > 0)[0]
    reshaped_indices = [(i // d, i % d) for i in flattened_indices]
    return reshaped_indices


def paths_containing_edge(l, row, position):
    # edge is ENTERING layer l
    # position is 0 or 1 (first or second coeff on row)
    e_ = np.zeros(2 * d * L)
    e_[2 * (l - 1) * d + 2 * row + position] = 1
    flattened_indices = np.where(P @ e_ > 0)[0]
    reshaped_indices = [(i // d, i % d) for i in flattened_indices]
    return reshaped_indices


def rank_activated_paths(activated_paths_list):
    indices_list = []
    for i in range(d):
        for j in range(d):
            for activated_paths in activated_paths_list:
                if activated_paths[i, j] == 1:
                    indices_list.append(i * d + j)
                    break
    indices_list.sort()
    indices_list = np.array(indices_list)
    P_activated = P[indices_list]
    return np.linalg.matrix_rank(P_activated)


# %% Sampling


def collect_jacobians(grid, forward, jacobian):
    jacobian_list = []
    x_list = []
    activated_paths_list = []
    activated_paths_mean = np.zeros((d, d))
    for x in grid:
        _, activations = forward(x, store_activations=True)
        jac = jacobian(activations)
        activated_paths = support_matrix(jac, threshold=1e-9)
        if np.max(activated_paths - activated_paths_mean) == 1:
            n_act = len(activated_paths_list)
            activated_paths_mean = (n_act * activated_paths_mean + activated_paths) / (
                n_act + 1
            )
            jacobian_list.append(jac)
            activated_paths_list.append(activated_paths)
            x_list.append(x)
    return jacobian_list, x_list, activated_paths_list


def merge_jacobians_exact(jacobian_list):
    # works only when the network is exactly butterfly
    jacobian_array = np.array(jacobian_list)
    jacobian_array[jacobian_array == 0] = -np.inf
    J_hat = np.max(jacobian_array, axis=0)
    J_hat[J_hat == -np.inf] = 0
    return J_hat


## uniform sampling
grid_scale = 1.0  # does not matter as forward is positively homogeneous
points_per_coordinate = 4
grid = create_grid(d, grid_scale, points_per_coordinate, type="edge_grid")
# len(grid) is points_per_coordinate ** d - (points_per_coordinate - 2) ** d

jacobian_list, x_list, activated_paths_list = collect_jacobians(grid, forward, jacobian)
J_hat = merge_jacobians_exact(jacobian_list)

print(
    "rank of activated paths grid:",
    rank_activated_paths(activated_paths_list),
    "full rank:",
    (L + 1) * d,
)

## random sampling
n_samples = len(grid)
print("n_samples:", n_samples, "d:", d)
random_grid = [np.random.randn(d) for _ in range(n_samples)]

jacobian_list, x_list, activated_paths_list = collect_jacobians(
    random_grid, forward, jacobian
)
print("retained samples,", len(jacobian_list))
J_hat = merge_jacobians_exact(jacobian_list)

print(
    "rank of activated paths random:",
    rank_activated_paths(activated_paths_list),
    "full rank:",
    (L + 1) * d,
)

## print('true product - estimated product:', product_of_weights(W) - J_hat)

# %% Factorization

W_hat = factorization_with_missing_values(J_hat, order="increasing")


# %% Correcting orientations


def correct_orientations(x_list, activated_paths_list, W_hat, verbose=False):

    checked_neurons = np.zeros((L - 1, d))

    for x, activated_paths in zip(x_list, activated_paths_list):
        for l in range(L - 1):
            x = W_hat[l] @ x
            activations_l = np.where(x > 0, 1, 0)
            for i in range(d):
                paths_containing_i = paths_containing_neuron(l, i)
                if activations_l[i] == 0:
                    for q, r in paths_containing_i:
                        if activated_paths[q, r] == 1:
                            if verbose:
                                print("changed sign with neuron", i, "in layer", l)
                            W_hat[l][i] *= -1
                            W_hat[l + 1][:, i] *= -1
                            checked_neurons[l, i] = 1
                            break
                elif activations_l[i] == 1:
                    for q, r in paths_containing_i:
                        if activated_paths[q, r] == 1:
                            checked_neurons[l, i] = 1
                            break
            x = ReLU(x)

    return W_hat, checked_neurons


W_hat, checked_neurons = correct_orientations(
    x_list, activated_paths_list, W_hat.copy(), verbose=False
)

# checking orientations
for l in range(L - 1):
    for i in range(d):
        if not np.all(W[l][i] * W_hat[l][i] >= 0):
            paths = paths_containing_neuron(l + 1, i)
            print("Warning: wrong orientation for neuron", i, "in layer", l + 1)
            break

regularization = 1e-10
num_tests = 100
diff_list = []
for n in range(num_tests):
    x = np.random.randn(d)
    y, _ = forward(x, W=W_hat)
    y_true, _ = forward(x)
    diff_list.append(
        np.linalg.norm(y - y_true) / (np.linalg.norm(y_true) + regularization)
    )

# compute the median
median_diff = np.median(diff_list)
print("for x random, median relative error in forward pass:", median_diff)
print("for x in grid, relative errors in forward pass:")
for x in x_list:
    print(
        (
            np.linalg.norm(forward(x, W=W_hat)[0] - forward(x)[0])
            / (np.linalg.norm(forward(x)[0]) + regularization)
        )
    )
