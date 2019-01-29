import os

import numpy as np
from scipy.signal import convolve2d
from torch import Tensor
from torch.utils.data import Dataset


def grid_distances(n, torus=False):
    """Returns n x n x n x n matrix of euclidean distances between
    pixels in two n x n images."""

    base_coord = np.arange(n)
    one_d_squared_distances = (base_coord[np.newaxis, :] - base_coord[:, np.newaxis]) ** 2

    if torus:
        loop_coord = np.concatenate((base_coord[n // 2:], base_coord[:n // 2]))
        one_d_squared_distances_loop = (loop_coord[np.newaxis, :] - loop_coord[:, np.newaxis]) ** 2
        one_d_squared_distances = np.minimum(one_d_squared_distances, one_d_squared_distances_loop)

    two_d_squared_distances = one_d_squared_distances[:, np.newaxis, :, np.newaxis] + one_d_squared_distances[np.newaxis, :, np.newaxis, :]
    return np.sqrt(two_d_squared_distances)


def rbe_kernel(d, length_scale, sigma):
    """Computes RBE kernel from distance matrix."""
    return sigma * np.exp(-d * d / (2 * length_scale * length_scale))


def sample_rbe_gp(size, length_scale, samples, torus = False):
    N = size * size
    D = grid_distances(size, torus)
    K = rbe_kernel(D, length_scale, 1).reshape(N, N)
    y = np.random.multivariate_normal(np.zeros(N), K, size=samples)
    Y = y.reshape(samples, size, size)
    return Y


def convolve2d_vectorized(X, filter, **kwargs):
    return np.concatenate([convolve2d(X[i], filter, **kwargs)[np.newaxis, :, :] for i in range(len(X))])


def gaussian_process_posterior(y, K, L, obs=None, fast = True):
    """
    Given a random variable x drawn from a GP with kernel K,
    noise n drawn from a GP with covariance L,
    and observation y = x + n, we compute the posterior distribution
    for x. If obs is set, then we only observe some of the
    entries of y.

    :param y: measurement
    :param K: covariance of x
    :param L: covariance of noise
    :param obs: boolean vector of which entries of y we observe
    :return: posterior mean mu, posterior covariance Sigma
    """

    n = len(y)
    if obs is None:
        obs = np.repeat(True, n)

    y_obs = y[obs]
    K_y = K[obs, :][:, obs] + L[obs, :][:, obs]

    if fast:
        mu = np.dot(K[:, obs], np.linalg.solve(K_y, y_obs))
        Sigma = K - np.dot(K[:, obs], np.linalg.solve(K_y, K[obs,:]))
    else:
        K_y_inv = np.linalg.inv(K_y)
        mu = np.dot(K[:, obs], np.dot(K_y_inv, y_obs))

        Sigma = K - np.dot(K[:, obs], np.dot(K_y_inv, K[obs, :]))

    return mu, Sigma


class GPDataset(Dataset):
    """Construct a dataset from a spatial Gaussian Process."""

    def __init__(self, size, length_scale, samples, noise_std, torus = False, **kwargs):
        self.noise_std = noise_std
        self.size = size
        self.length_scale = length_scale
        self.samples = samples

        N = size * size
        self.N = N

        self.torus = torus
        D = grid_distances(size, torus)

        self.K = rbe_kernel(D, length_scale, 1).reshape(N, N)
        self.L = np.eye(N) * noise_std ** 2

        y = np.random.multivariate_normal(np.zeros(N), self.K, size=samples)
        s = np.random.multivariate_normal(np.zeros(N), self.L, size=samples)
        x = y + s

        s2 = np.random.multivariate_normal(np.zeros(N), self.L, size=samples)
        x2 = y + s2

        self.X = x.reshape(samples, size, size)
        self.Y = y.reshape(samples, size, size)
        self.S = s.reshape(samples, size, size)

        self.X2 = x2.reshape(samples, size, size)
        self.S2 = s2.reshape(samples, size, size)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return (Tensor(self.X[index]).unsqueeze(0),
                Tensor(self.X2[index]).unsqueeze(0),
                Tensor(self.Y[index]).unsqueeze(0))

    def compute_posterior(self):
        x = self.X.reshape(self.samples, self.N).transpose()

        K_y = self.K + self.L
        K_y_inv = np.linalg.inv(K_y)

        mu = np.dot(self.K, np.dot(K_y_inv, x))

        self.Mu = mu.transpose().reshape(self.samples, self.size, self.size)
        return self.Mu

    def compute_deconv_kernel(self):
        inverse = np.dot(self.K, np.linalg.inv(self.K + self.L))

        deconv_kernel = inverse.reshape(self.size, self.size, self.size, self.size)[self.size // 2, self.size // 2]

        # center the center in the kernel
        if self.size % 2 == 0:
            deconv_kernel = deconv_kernel[1:, 1:]

        return deconv_kernel

    def conv_denoise(self):
        deconv_kernel = self.compute_deconv_kernel()
        boundary = 'wrap' if self.torus else 'symm'
        return convolve2d_vectorized(self.X, deconv_kernel, mode='same', boundary=boundary)

    def compute_deconv_mask_kernel(self, noise=True):
        def grid_to_list(i, j, n):
            return n * i + j

        center = self.size // 2

        i = grid_to_list(center, center, self.size)
        obs = np.repeat(True, self.N)
        obs[i] = False

        if noise:
            deconv_mask = np.dot(self.K[:, obs], np.linalg.inv(self.K[obs, :][:, obs] + self.L[obs, :][:, obs]))
        else:
            deconv_mask = np.dot(self.K[:, obs], np.linalg.inv(self.K[obs, :][:, obs] + 1e-4*np.eye(len(self.K) - 1)))

        deconv_mask = np.insert(deconv_mask, i, 0, axis=1)
        deconv_mask_kernel = deconv_mask.reshape(self.size, self.size, self.size, self.size)[center, center]

        # center the center in the kernel
        if self.size % 2 == 0:
            deconv_mask_kernel = deconv_mask_kernel[1:, 1:]

        return deconv_mask_kernel

    def conv_denoise_mask(self, noise=True):
        deconv_mask_kernel = self.compute_deconv_mask_kernel(noise)
        boundary = 'wrap' if self.torus else 'symm'

        if noise:
            return convolve2d_vectorized(self.X, deconv_mask_kernel, mode='same', boundary=boundary)
        else:
            return convolve2d_vectorized(self.Y, deconv_mask_kernel, mode='same', boundary=boundary)


def make_test_gp_dataset():
    seed = 2018
    np.random.seed(seed)

    size = 32
    train_samples = 1024
    val_samples = 128
    test_samples = 128
    noise_std = 0.5
    length_scale = 2

    train_data = GPDataset(size, length_scale, train_samples, noise_std)
    val_data = GPDataset(size, length_scale, val_samples, noise_std)
    test_data = GPDataset(size, length_scale, test_samples, noise_std)

    # Notation is confusing here because Y = X + S is the model, but the neural net has input Y and target X.
    # We insert a channel axis.

    X_train = train_data.X[:, np.newaxis, :, :]
    X_val = val_data.X[:, np.newaxis, :, :]

    X_test = test_data.X[:, np.newaxis, :, :]
    Y_test = test_data.Y[:, np.newaxis, :, :]
    Mu_test = test_data.compute_posterior()[:, np.newaxis, :, :]
    Mu_mask_test = test_data.conv_denoise_mask()[:, np.newaxis, :, :]

    os.makedirs('data/gp_small', exist_ok=True)
    with open('data/gp_small/train.npz', 'wb') as outfile:
        np.savez(outfile, X=X_train)
    with open('data/gp_small/val.npz', 'wb') as outfile:
        np.savez(outfile, X=X_val)
    with open('data/gp_small/test.npz', 'wb') as outfile:
        np.savez(outfile, X=X_test, Y=Y_test, Mu=Mu_test, Mu_mask=Mu_mask_test)

    def MSE(x, y, pad):
        x = x[:, :, pad:-pad, pad:-pad]
        y = y[:, :, pad:-pad, pad:-pad]
        return np.mean((x - y) ** 2)

    # Check that the posteriors are better than the noise, and that the masking is worse than not.

    X_mse = MSE(X_test, Y_test, 2)
    Mu_mse = MSE(Mu_test, Y_test, 2)
    Mu_mask_mse = MSE(Mu_mask_test, Y_test, 2)
    assert X_mse > Mu_mask_mse
    assert Mu_mask_mse > Mu_mse
