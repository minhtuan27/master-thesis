import tensorflow as tf
import numpy as np

from numpy.random import multivariate_normal
from scipy.stats import wishart

def Normal_Wishart(mu_0, lamb, W, nu, seed=None):
    """Function extracting a Normal_Wishart random variable"""
    # first draw a Wishart distribution:
    Lambda = wishart(df=nu, scale=W, seed=seed).rvs()  # NB: Lambda is a matrix.
    # then draw a Gaussian multivariate RV with mean mu_0 and(lambda*Lambda)^{-1} as covariance matrix.
    cov = np.linalg.inv(lamb * Lambda) if (lamb * Lambda).ndim == 2 else np.array([[1. / (lamb * Lambda)]])
    mu = multivariate_normal(mu_0, cov)
    return mu, Lambda, cov

class BPMF(tf.Module):
    """
    Apply the Bayesian Probabilistic Matrix Factorization algorithm.
    Note: In practice, we found that it is more efficient to use numpy to compute the updates for C and X.
        
    Input:
        train_tensor: tf.Tensor of shape (m, n), where m is the number of movies and n is the number of users
        train_M: tf.Tensor of shape (m, n), the mask tensor
        num_factors: int, the number of factors
        init_scale: float, the scale of the initialization
        mu_0: np.array of shape (num_factors,), the prior mean for the normal distribution
        Beta_0: float, the prior weight for the normal distribution
        nu_0: float, the prior degrees of freedom for the Wishart distribution
        W_0: np.array of shape (num_factors, num_factors), the prior scale matrix for the Wishart distribution
    Output:
        C: tf.Variable of shape (m, num_factors)
        X: tf.Variable of shape (num_factors, n)
    """

    def __init__(self, m, n, num_factors, init_scale,
                 mu_0=None, Beta_0=None, W_0=None, nu_0=None):
        super(BPMF, self).__init__()

        self.C = tf.Variable(tf.multiply(tf.random.truncated_normal([m, num_factors], mean=1.0, stddev=0.02), init_scale))
        self.X = tf.Variable(tf.multiply(tf.random.truncated_normal([num_factors, n], mean=1.0, stddev=0.02), init_scale))
        
        self.num_factors = num_factors
        self.m = m
        self.n = n
        
        self.mu_0 = mu_0 or np.zeros(num_factors)
        self.Beta_0 = Beta_0 or 2
        self.nu_0 = nu_0 or self.num_factors
        self.W_0 = W_0 or np.eye(self.num_factors)

        self.Lambda_C = np.eye(self.num_factors)
        self.Lambda_X = np.eye(self.num_factors)
        self.alpha = 2

    def __call__(self, train_tensor):
        Y = train_tensor.numpy()
        prev_X = self.X.numpy()
        prev_C_transpose = self.C.numpy().T

        alpha = 2

        Beta_0_star = self.Beta_0 + self.m
        nu_0_star = self.nu_0 + self.m
        W_0_inv = np.linalg.inv(self.W_0)

        X_average = np.sum(prev_X, axis=1) / self.m
        S_bar_X = np.dot(prev_X, np.transpose(prev_X)) / self.m
        mu_0_star_X = (self.Beta_0 * self.mu_0 + self.m * X_average) / (self.Beta_0 + self.m)
        W_0_star_X_inv = W_0_inv + self.m * S_bar_X + self.Beta_0 * self.m / (self.Beta_0 + self.m) * np.dot(
            np.transpose(np.array(self.mu_0 - X_average, ndmin=2)), np.array((self.mu_0 - X_average), ndmin=2))
        W_0_star_X = np.linalg.inv(W_0_star_X_inv)
        mu_X, Lambda_X, cov_X = Normal_Wishart(mu_0_star_X, Beta_0_star, W_0_star_X, nu_0_star, seed=None)

        C_average = np.sum(prev_C_transpose, axis=1) / self.m
        S_bar_C = np.dot(prev_C_transpose, np.transpose(prev_C_transpose)) / self.m
        mu_0_star_C = (self.Beta_0 * self.mu_0 + self.m * C_average) / (self.Beta_0 + self.m)
        W_0_star_C_inv = W_0_inv + self.m * S_bar_C + self.Beta_0 * self.m / (self.Beta_0 + self.m) * np.dot(
            np.transpose(np.array(self.mu_0 - C_average, ndmin=2)), np.array((self.mu_0 - C_average), ndmin=2))
        W_0_star_C = np.linalg.inv(W_0_star_C_inv)
        mu_C, Lambda_C, cov_C = Normal_Wishart(mu_0_star_C, Beta_0_star, W_0_star_C, nu_0_star, seed=None)

        C_transpose = np.array([])
        X = np.array([])

        for i in range(self.m):
            Lambda_C_2 = np.zeros((self.num_factors, self.num_factors))
            mu_i_star_1 = np.zeros(self.num_factors)
            for j in range(self.n):
                if Y[i, j] != 0:
                    Lambda_C_2 = Lambda_C_2 + np.dot(np.transpose(np.array(prev_X[:, j], ndmin=2)),
                                                        np.array((prev_X[:, j]), ndmin=2))
                    mu_i_star_1 = prev_X[:, j] * Y[i, j] + mu_i_star_1

            Lambda_i_star_C = Lambda_C + alpha * Lambda_C_2
            Lambda_i_star_C_inv = np.linalg.inv(Lambda_i_star_C)

            mu_i_star_part = alpha * mu_i_star_1 + np.dot(Lambda_C, mu_C)
            mu_i_star = np.dot(Lambda_i_star_C_inv, mu_i_star_part)
            C_transpose = np.append(C_transpose, multivariate_normal(mu_i_star, Lambda_i_star_C_inv))

        C_transpose = np.transpose(np.reshape(C_transpose, (self.m, self.num_factors)))

        for j in range(self.n):
            Lambda_X_2 = np.zeros((self.num_factors, self.num_factors))
            mu_i_star_1 = np.zeros(self.num_factors)
            for i in range(self.m):
                if Y[i, j] != 0:
                    Lambda_X_2 = Lambda_X_2 + np.dot(np.transpose(np.array(C_transpose[:, i], ndmin=2)),
                                                        np.array((C_transpose[:, i]), ndmin=2))
                    mu_i_star_1 = C_transpose[:, i] * Y[i, j] + mu_i_star_1

            Lambda_j_star_X = Lambda_X + alpha * Lambda_X_2
            Lambda_j_star_X_inv = np.linalg.inv(Lambda_j_star_X)

            mu_i_star_part = alpha * mu_i_star_1 + np.dot(Lambda_X, mu_X)
            mu_j_star = np.dot(Lambda_j_star_X_inv, mu_i_star_part)
            X = np.append(X, multivariate_normal(mu_j_star, Lambda_j_star_X_inv))

        X = np.transpose(np.reshape(X, (self.n, self.num_factors)))

        self.C.assign(C_transpose.T)
        self.X.assign(X)