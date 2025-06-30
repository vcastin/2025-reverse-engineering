import numpy as np
from src.utils import partial_prod_butterfly_supports


def ReLU(x):
    return np.maximum(0, x)


def support_matrix(W, threshold=0):
    return np.where(abs(W) > threshold, 1, 0)


def split_product(
    jacobian, S1, S2, d_in, scaling=1e2, max_iter=20, verbose=False, factor=2
):
    # pick an orthogonal matrix
    O = scaling * np.linalg.qr(np.random.randn(d_in, d_in))[0]
    for i in range(d_in):
        if verbose:
            print(f"Test with vector {i}")
        x = O[:, i]
        for _ in range(max_iter):
            J1 = jacobian(x)
            J2 = jacobian(-x)
            full_support = S2 @ S1
            if np.allclose(support_matrix(J1 + J2), full_support) and np.allclose(
                support_matrix(J1 - J2), full_support
            ):
                return J1, J2, x
            x = factor * x
    if not np.allclose(support_matrix(J1 + J2), full_support):
        print("WARNING: one component could not be activated")
    if not np.allclose(support_matrix(J1 - J2), full_support):
        print("WARNING: one component could not be deactivated")
    return J1, J2, x


def activated_neurons(J, S1, S2, H):
    def rank_one_component(i, S1, S2):
        return np.outer(S2[:, i], S1[i, :])

    activations = np.zeros(H)
    for i in range(H):
        if not np.allclose(J * rank_one_component(i, S1, S2), np.zeros_like(J)):
            activations[i] = 1
    return activations


def correct_orientations(W1_hat, x, J1, S1, S2, H):
    product = W1_hat @ x
    predicted_activations = (product > 0).astype(int)
    real_activations = activated_neurons(J1, S1, S2, H)
    orientations = 2 * (predicted_activations == real_activations).astype(int) - 1
    return orientations


def find_biases_dichotomy(jacobian, W1_hat, x, J1, S1, S2, H, verbose=False):
    def dichotomy(x, i, threshold=1e-6, max_iter=1e3):
        x_activates_i = activated_neurons(J1, S1, S2, H)[i]
        if x_activates_i:
            t_deactivate, t_activate = -1, 1
        else:
            t_deactivate, t_activate = 1, -1
        iteration = 0
        while abs(t_activate - t_deactivate) > threshold and iteration < max_iter:
            iteration += 1
            t = (t_activate + t_deactivate) / 2
            x_t = t * x
            J_t = jacobian(x_t)
            activations = activated_neurons(J_t, S1, S2, H)
            if activations[i] == 1:
                t_activate = t
            else:
                t_deactivate = t
        if verbose:
            print("t_activate = ", t_activate, ", t_deactivate = ", t_deactivate)
        return t

    b_hat = np.zeros(H)
    for i in range(H):
        t = dichotomy(x, i)
        b_hat[i] = -np.inner(W1_hat[i, :], x) * t

    return b_hat


def find_biases_neighbors(
    forward, jacobian, W2_hat, x, J1, J2, S1, S2, H, verbose=False
):

    def check_neighbors(Jx, Jy):
        diff_activations = abs(
            activated_neurons(Jx, S1, S2, H) - activated_neurons(Jy, S1, S2, H)
        )
        if diff_activations.sum() == 1:
            return np.where(diff_activations == 1)[0][
                0
            ]  # index of the neuron that switches activation between x and y
        elif diff_activations.sum() == 0:
            return -1  # no neuron switches activation between x and y
        return -2  # more than one neuron switches activation between x and y

    def compute_bias(x, Jx, y, Jy, W2_hat):
        # takes two j-neighbors x and y, and returns the bias of neuron j
        j = check_neighbors(Jx, Jy)
        if j < 0:
            print("WARNING: compute_bias should take neighbors as input")
            return None
        if activated_neurons(Jx, S1, S2, H)[j] == 0:
            x, Jx, y, Jy = y, Jy, x, Jx
        X = Jx @ x - Jy @ y + forward(y) - forward(x)
        i = 0
        while W2_hat[i, j] == 0:
            i += 1
        b_j = -X[i] / W2_hat[i, j]
        return b_j

    sample_neighbors_iteration = np.array([0])
    found_biases = np.zeros(H, dtype=bool)
    biases = np.zeros(H)

    def sample_neighbors(neighbors, jacobians, W2_hat, max_iter=1e4):
        # neighbors is a couple of points, with their jacobians stored in jacobians
        # the function modifies b_hat
        sample_neighbors_iteration[0] = sample_neighbors_iteration[0] + 1
        if sample_neighbors_iteration[0] > max_iter:
            raise ValueError("sample_neighbors reached max_iter")

        x, y = neighbors
        Jx, Jy = jacobians
        j = check_neighbors(Jx, Jy)
        if j >= 0:
            biases[j] = compute_bias(x, Jx, y, Jy, W2_hat)
            found_biases[j] = True
        elif j == -2:  # then at least one bias is missing
            z = (x + y) * 0.5
            Jz = jacobian(z)
            sample_neighbors([x, z], [Jx, Jz], W2_hat)
            sample_neighbors([z, y], [Jz, Jy], W2_hat)

    neighbors = (x, -x)
    jacobians = (J1, J2)
    sample_neighbors(neighbors, jacobians, W2_hat)
    if verbose:
        print("Found biases:", found_biases)
    return biases


# For butterfly reconstruction


def store_index_of_components(support1, support2):
    d_out, H = support2.shape
    H_, d_in = support1.shape
    assert H == H_
    indices = np.zeros((d_out, d_in), dtype=int)
    for h in range(H):
        indices += h * np.outer(support2[:, h], support1[h, :]).astype(int)
    return indices


def factorization_with_missing_values(J_hat, order="increasing"):

    L = int(np.log2(J_hat.shape[0]))
    assert 2**L == J_hat.shape[0]

    to_factorize = J_hat
    W_hat = []

    for cut_index in range(1, L):
        if order == "increasing":
            support2 = partial_prod_butterfly_supports(
                L, cut_index, L
            )  # S_L ... S_{cut_index + 1}
            support1 = partial_prod_butterfly_supports(
                L, cut_index - 1, cut_index
            )  # S_{cut_index}
        elif order == "decreasing":
            support2 = partial_prod_butterfly_supports(
                L, L - cut_index, L - cut_index + 1
            )
            support1 = partial_prod_butterfly_supports(L, 0, L - cut_index)

        indices = store_index_of_components(support1, support2)
        constraints = np.copy(to_factorize)
        # print('constraints', constraints)
        W1_hat, W2_hat = np.zeros_like(support1), np.zeros_like(support2)
        d_in, d_out = support1.shape[1], support2.shape[0]

        def update_weights_and_constraints(num_factor, i, j, value_ij):
            if value_ij == 0:
                raise ValueError("value_ij should be nonzero")
            if num_factor == 1:
                for row in range(d_out):
                    h = indices[row, j]
                    if h == i and constraints[row, j] != 0:
                        if W2_hat[row, i] != 0:
                            assert (
                                abs(W2_hat[row, i] - constraints[row, j] / value_ij)
                                < 1e-9
                            )
                        W2_hat[row, i] = constraints[row, j] / value_ij
                        constraints[row, j] = 0
                        update_weights_and_constraints(2, row, i, W2_hat[row, i])
            elif num_factor == 2:
                for col in range(d_in):
                    h = indices[i, col]
                    if h == j and constraints[i, col] != 0:
                        if W1_hat[j, col] != 0:
                            assert (
                                abs(W1_hat[j, col] - constraints[i, col] / value_ij)
                                < 1e-9
                            )
                        W1_hat[j, col] = constraints[i, col] / value_ij
                        constraints[i, col] = 0
                        update_weights_and_constraints(1, j, col, W1_hat[j, col])
            else:
                raise ValueError("num_factor should be 1 or 2")

        for i in range(d_out):
            for j in range(d_in):
                if constraints[i, j] != 0:
                    h = indices[i, j]
                    assert W1_hat[h, j] == 0
                    W1_hat[h, j] = np.random.randn()
                    update_weights_and_constraints(1, h, j, W1_hat[h, j])
        if order == "increasing":
            W_hat.append(W1_hat)
            to_factorize = W2_hat
        elif order == "decreasing":
            W_hat.append(W2_hat)
            to_factorize = W1_hat

    W_hat.append(to_factorize)
    if order == "decreasing":
        W_hat.reverse()
    return W_hat


def create_grid(d, grid_scale, points_per_coordinate, type="edge_grid"):
    coordinate_values = grid_scale * np.linspace(-1, 1, points_per_coordinate)
    grid_arrays = np.meshgrid(*([coordinate_values] * d))
    grid_points = np.stack(grid_arrays, axis=-1).reshape(-1, d)

    if type == "full_grid":
        grid = [np.array(point) for point in grid_points]
    elif type == "edge_grid":
        grid = [
            np.array(point)
            for point in grid_points
            if any(abs(coord) == grid_scale for coord in point)
        ]
    return grid
