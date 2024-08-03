import tensorflow as tf

class BernoulliDF(tf.Module):
    """
    Apply the Dictionary Filter algorithm with Bernoulli distribution model.
    
    Input:
        Y: tf.Tensor of shape (m, d), where m is the number of samples and d is the batch size
    Variables:
        C: tf.Tensor of shape (m, r)
        V: tf.Tensor of shape (r, r)
    Output:
        X: tf.Tensor of shape (r, d)
    """

    def __init__(self, m, n, r, gamma_X, gamma_C, sigma_C, name=None):
        super().__init__(name=name)
        self.C = tf.Variable(tf.random.truncated_normal([m, r], mean=0.0, stddev=1.0))
        self.X = tf.Variable(tf.random.truncated_normal([r, n], mean=0.0, stddev=1.0))
        self.gamma_X = tf.constant(gamma_X, dtype=tf.float32)
        self.gamma_C = tf.constant(gamma_C, dtype=tf.float32)
        self.sigma_C = tf.constant(sigma_C ** 2, dtype=tf.float32)

    @tf.function
    def gradient(self, Y, C, X):
        return tf.subtract(
            tf.multiply(Y, tf.subtract(1.0, tf.sigmoid(tf.matmul(C, X)))),
            tf.multiply(tf.subtract(1.0, Y), tf.sigmoid(tf.matmul(C, X)))
        )

    @tf.function
    def __call__(self, Yk, start_index):
        end_index = tf.add(start_index, tf.shape(Yk)[1])
        Xk = self.X[:, start_index:end_index]

        # Update X
        Xk = tf.add(
            Xk,
            tf.multiply(
                self.gamma_X,
                tf.matmul(self.C, self.gradient(Yk, self.C, Xk), transpose_a=True)
            )
        )
        self.X[:, start_index:end_index].assign(Xk)

        # Update C
        self.C.assign(
            tf.add(
                self.C,
                tf.multiply(
                    self.gamma_C,
                    tf.subtract(
                        tf.matmul(self.gradient(Yk, self.C, Xk), Xk, transpose_b=True),
                        tf.divide(self.C, self.sigma_C)
                    )
                )
            )
        )

        return tf.sigmoid(tf.matmul(self.C, Xk))