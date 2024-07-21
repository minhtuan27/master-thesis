import tensorflow as tf

class NMF(tf.Module):
    """
    Apply the Non-negative Matrix Factorization algorithm.

    Input:
        train_tensor: tf.Tensor of shape (m, n), where m is the number of movies and n is the number of users
        train_M: tf.Tensor of shape (m, n), the mask tensor
        num_factors: int, the number of factors
        init_scale: float, the scale of the initialization
    Output:
        C: tf.Variable of shape (m, num_factors)
        X: tf.Variable of shape (num_factors, n)
    """

    def __init__(self, m, n, num_factors, init_scale):
        super().__init__()

        self.num_factors = num_factors

        self.C = tf.Variable(tf.multiply(tf.random.truncated_normal([m, num_factors], mean=1.0, stddev=0.02), init_scale))
        self.X = tf.Variable(tf.multiply(tf.random.truncated_normal([num_factors, n], mean=1.0, stddev=0.02), init_scale))

    @tf.function
    def __call__(self, train_tensor, train_M):
        # Update user factors
        self.C.assign(tf.multiply(
            self.C, 
            tf.math.divide_no_nan(
                tf.matmul(train_tensor, tf.transpose(self.X)),
                tf.matmul(tf.multiply(train_M, tf.matmul(self.C, self.X)), tf.transpose(self.X)))))
        
        # Update movie factors
        self.X.assign(tf.multiply(
            self.X, 
            tf.math.divide_no_nan(
                tf.matmul(tf.transpose(self.C), train_tensor),
                tf.matmul(tf.transpose(self.C), tf.multiply(train_M, tf.matmul(self.C, self.X))))))