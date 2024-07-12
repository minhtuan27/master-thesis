import tensorflow as tf

class MVGMF(tf.Module):
    """
    Apply the Multiview Gaussian Matrix Factorization algorithm.

    Input:
        train_tensor: tf.Tensor of shape (m, n), where m is the number of movies and n is the number of users
        train_M: tf.Tensor of shape (m, n), the mask tensor
        num_factors: int, the number of factors
        k: float, the regularization parameter
        sigma: float, the noise parameter
        init_scale: float, the scale of the initialization
    Output:
        C: tf.Variable of shape (m, num_factors)
        X: tf.Variable of shape (num_factors, n)
        V: tf.Variable of shape (num_factors, num_factors)
        W: tf.Variable of shape (num_factors, num_factors)
    """

    def __init__(self, m, n, num_factors, k, sigma, init_scale):
        super().__init__()

        self.m = m
        self.n = n
        self.num_factors = num_factors
        self.sigmasq = sigma ** 2
        
        self.C = tf.Variable(tf.multiply(tf.random.truncated_normal([m, num_factors], mean=1.0, stddev=0.02), init_scale))
        self.X = tf.Variable(tf.multiply(tf.random.truncated_normal([num_factors, n], mean=1.0, stddev=0.02), init_scale))
        self.V = tf.Variable(tf.multiply(tf.eye(num_factors, dtype=tf.float32), k))
        self.W = tf.Variable(tf.multiply(tf.eye(num_factors, dtype=tf.float32), k))

    @tf.function
    def __call__(self, train_tensor, train_M):
        # temp_cv = VX(\sigma^2I+X^TVX)^-1
        temp_cv = tf.matmul(
            tf.matmul(self.V, self.X),
            tf.linalg.pinv(
                tf.add(
                    tf.multiply(tf.eye(self.n), self.sigmasq),
                    tf.matmul(tf.transpose(self.X), tf.matmul(self.V, self.X))
                )
            )
        )

        # C = C + (M*(Y-CX))(temp_cv^T)
        self.C.assign_add(
            tf.matmul(
                tf.multiply(train_M, tf.subtract(train_tensor, tf.matmul(self.C, self.X))),
                tf.transpose(temp_cv)
            )
        )

        # V = V - temp_cv X^TV
        self.V.assign_sub(
            tf.matmul(
                temp_cv,
                tf.matmul(tf.transpose(self.X), self.V)
            )
        )

        # temp_xw = WC^T(\sigma^2I+CWC^T)^-1
        temp_xw = tf.matmul(
            tf.matmul(self.W, tf.transpose(self.C)),
            tf.linalg.pinv(
                tf.add(
                    tf.multiply(tf.eye(self.m), self.sigmasq),
                    tf.matmul(tf.matmul(self.C, self.W), tf.transpose(self.C))
                )
            )
        )

        # X = X + temp_xw(M*(Y-CX))
        self.X.assign_add(
            tf.matmul(
                temp_xw,
                tf.multiply(train_M, tf.subtract(train_tensor, tf.matmul(self.C, self.X)))
            )
        )

        # W = W - temp_xw CW
        self.W.assign_sub(
            tf.matmul(
                temp_xw,
                tf.matmul(self.C, self.W)
            )
        )