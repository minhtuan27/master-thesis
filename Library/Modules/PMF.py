import tensorflow as tf
import numpy as np

class PMF(tf.Module):
    """
    Apply the Probabilistic Matrix Factorization algorithm.
    Note: In practice, we found that it is more efficient to use numpy to compute the updates for C and X.
        
    Input:
        train_tensor: tf.Tensor of shape (m, n), where m is the number of movies and n is the number of users
        train_M: tf.Tensor of shape (m, n), the mask tensor
        num_factors: int, the number of factors
        init_scale: float, the scale of the initialization
        lambda_C: float, the regularization parameter for movie factors
        lambda_X: float, the regularization parameter for user factors
    Output:
        C: tf.Variable of shape (m, num_factors)
        X: tf.Variable of shape (num_factors, n)
    """

    def __init__(self, m, n, num_factors, init_scale, lambda_C, lambda_X):
        super().__init__()

        self.m = m
        self.n = n
        self.num_factors = num_factors

        self.C = tf.Variable(tf.multiply(tf.random.truncated_normal([m, num_factors], mean=1.0, stddev=0.02), init_scale))
        self.X = tf.Variable(tf.multiply(tf.random.truncated_normal([num_factors, n], mean=1.0, stddev=0.02), init_scale))

        self.lambda_C = lambda_C
        self.lambda_X = lambda_X
    
    def __call__(self, train_tensor):
        Y = train_tensor.numpy()
        C = self.C.numpy()
        X = self.X.numpy()

        for i in range(self.m):
            X_j = X[:, Y[i, :] > 0]
            C[i, :] = np.dot(np.linalg.inv(np.dot(X_j, X_j.T) + self.lambda_C * np.identity(self.num_factors)), np.dot(Y[i, Y[i, :] > 0], X_j.T))
            
        for j in range(self.n):
            C_i = C[Y[:, j] > 0, :]
            X[:, j] = np.dot(np.linalg.inv(np.dot(C_i.T, C_i) + self.lambda_X * np.identity(self.num_factors)), np.dot(Y[Y[:, j] > 0, j], C_i))

        self.C.assign(C)
        self.X.assign(X)