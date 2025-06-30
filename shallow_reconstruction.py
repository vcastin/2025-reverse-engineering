# %%
import numpy as np
from src.utils import partial_prod_butterfly_supports
from src.hierarchical_fact import lifting_two_layers_factorization
from toolbox import (
    ReLU,
    split_product,
    activated_neurons,
    correct_orientations,
    find_biases_neighbors,
)
import matplotlib.pyplot as plt
import os

if not os.path.exists("figures"):
    os.makedirs("figures")

np.random.seed(91)

# %% defining the shallow network

## option 1 (uncomment to use)
# d_in, H, d_out = 2, 3, 2  # H is the hidden dimension
# S1 = np.array([[1,0], [0,1], [0,1]])  # support of W1
# S2 = np.array([[1, 0, 1], [1, 1, 0]]) # support of W2
## end of option 1

## option 2 (comment out to use option 1)
L = 4
d_in = 2**L
d_out = 2**L
H = 2**L
cut_index = 2
S2 = partial_prod_butterfly_supports(L, cut_index, L)  # S_L ... S_{cut_index}
S1 = partial_prod_butterfly_supports(L, 0, cut_index)
## end of option 2

W1 = np.random.randn(H, d_in) * S1
W2 = np.random.randn(d_out, H) * S2
b = np.random.randn(H)

forward_samples = []


def forward(x, track_samples=True):
    if track_samples:
        forward_samples.append(x)
    h = ReLU(W1 @ x + b)
    return W2 @ h


def activation_matrix(x):
    h = ReLU(W1 @ x + b)
    return np.diag(np.where(h > 0, 1, 0))


jacobian_samples = []


def jacobian(x, track_samples=True):
    if track_samples:
        jacobian_samples.append(x)
    return W2 @ activation_matrix(x) @ W1


def border_neuron_2d(h, x_lim, y_lim, n_samples):
    z = -W1[h] * b[h] / np.sum(W1[h] ** 2)
    v = np.zeros_like(z)
    v[0] = W1[h][1]
    v[1] = -W1[h][0]
    if np.linalg.norm(v) != 0:
        v /= np.linalg.norm(v)
        return (
            z[0] + np.linspace(-x_lim, x_lim, n_samples) * v[0],
            z[1] + np.linspace(-y_lim, y_lim, n_samples) * v[1],
        )
    else:
        return None, None


# %% reverse-engineering

## reconstructing the weights W1, W2
J1, J2, x = split_product(jacobian, S1, S2, d_in, verbose=False, factor=2.0)
print("x activates", activated_neurons(J1, S1, S2, H))
print("-x activates", activated_neurons(J2, S1, S2, H))

W2_hat, W1_hat = lifting_two_layers_factorization(S2, S1, J1 + J2)

orientations = correct_orientations(W1_hat, x, J1, S1, S2, H)
W2_hat = W2_hat * orientations
W1_hat = (W1_hat.T * orientations).T

jacobian_samples_for_weights = jacobian_samples.copy()

## reconstructing the biases
jacobian_samples = []

biases = find_biases_neighbors(forward, jacobian, W2_hat, x, J1, J2, S1, S2, H)

jacobian_samples_for_biases = jacobian_samples.copy()
forward_samples_for_biases = (
    forward_samples.copy()
)  # no forward samples needed for reconstructing the weights

## check
n_test_samples = 1000
maximal_difference = 0.0
for _ in range(n_test_samples):
    x_check = np.random.randn(d_in)
    norm_of_difference = np.linalg.norm(
        forward(x_check) - W2_hat @ ReLU(W1_hat @ x_check + biases)
    )
    maximal_difference = max(maximal_difference, norm_of_difference)
print(
    f"Largest error in reconstructed output for {n_test_samples} samples:",
    maximal_difference,
)

# %% Plot of the samples

sample_list_neighbors = np.array(
    jacobian_samples_for_weights + jacobian_samples_for_biases
)

f, ax = plt.subplots(1, 1, figsize=(3.5, 2))
ax.scatter(
    sample_list_neighbors[:, 0],
    sample_list_neighbors[:, 1],
    s=20,
    marker="+",
    alpha=0.8,
)
x_lim = np.max(np.abs(sample_list_neighbors[:, 0])) * 1.1
y_lim = np.max(np.abs(sample_list_neighbors[:, 1])) * 1.1
for h in range(H):
    x, y = border_neuron_2d(h, 2 * x_lim, 2 * y_lim, 100)
    if x is not None:
        ax.plot(x, y, color="black", linewidth=0.5)
ax.set_xlim(-x_lim, x_lim)
ax.set_ylim(-y_lim, y_lim)
plt.tight_layout()
ax.set_xticks([])
ax.set_yticks([])
plt.savefig("figures/samples_shallow_case.pdf")
