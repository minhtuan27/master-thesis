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

    def __init__(self, m, n, r, gamma_X, sigma_X, gamma_C, sigma_C, density, name=None):
        super().__init__(name=name)
        
        self.C = tf.Variable(tf.random.truncated_normal([m, r], mean=0.0, stddev=sigma_C, dtype=tf.float64))
        self.X = tf.Variable(tf.random.truncated_normal([r, n], mean=0.0, stddev=sigma_X, dtype=tf.float64))
        
        self.gamma_X = tf.constant(gamma_X, dtype=tf.float64)
        self.sigma_X = tf.constant(sigma_X ** 2, dtype=tf.float64)
        self.gamma_C = tf.constant(gamma_C, dtype=tf.float64)
        self.sigma_C = tf.constant(sigma_C ** 2, dtype=tf.float64)
        self.density = tf.constant(density, dtype=tf.float64)

        self.one = tf.constant(1.0, dtype=tf.float64)
        self.two = tf.constant(2.0, dtype=tf.float64)
        self.four = tf.constant(4.0, dtype=tf.float64)

    @tf.function
    def gradient(self, Y, C, X):
        return tf.subtract(
            tf.multiply(Y, tf.subtract(self.one, tf.sigmoid(tf.matmul(C, X)))),
            tf.multiply(tf.multiply(tf.subtract(self.one, Y), tf.sigmoid(tf.matmul(C, X))), self.density)
        )
    
    @tf.function
    def gradient_log_pi(self, Y, C, X):
        return tf.multiply(
            self.gamma_C,
            tf.subtract(
                tf.matmul(self.gradient(Y, C, X), X, transpose_b=True),
                tf.divide(self.C, self.sigma_C)
            )
        )
    
    # @tf.function
    # def log_pi(self, Y, C, X):
    #     return tf.subtract(
    #         tf.reduce_sum(
    #             tf.add(
    #                 tf.multiply(Y, tf.math.log_sigmoid(tf.matmul(C, X))),
    #                 tf.multiply(tf.subtract(self.one, Y), tf.math.log_sigmoid(tf.negative(tf.matmul(C, X))))
    #             )
    #         ),
    #         tf.divide(tf.square(tf.norm(C)), tf.multiply(self.two, self.sigma_C))
    #     )
    
    # @tf.function
    # def log_q(self, Y, X, C_tilde, C):
    #     return tf.negative(
    #         tf.divide(
    #             tf.square(tf.norm(tf.subtract(C_tilde, tf.add(C, self.gradient_log_pi(Y, C, X))))),
    #             tf.multiply(self.four, self.gamma_C)
    #         )
    #     )
    
    # @tf.function
    # def metropolis_hastings(self, Y, X, C_tilde, C):
    #     # print(self.log_pi(Y, C_tilde, X))
    #     # print(self.log_q(Y, X, C, C_tilde))
    #     # print(self.log_pi(Y, C, X))
    #     # print(self.log_q(Y, X, C_tilde, C))
    #     return tf.minimum(
    #         self.one,
    #         tf.math.exp(
    #             tf.subtract(
    #                 tf.add(self.log_pi(Y, C_tilde, X), self.log_q(Y, X, C, C_tilde)),
    #                 tf.add(self.log_pi(Y, C, X), self.log_q(Y, X, C_tilde, C))
    #             )
    #         )
    #     )

    @tf.function
    def __call__(self, Yk, start_index):
        end_index = tf.add(start_index, tf.shape(Yk)[1])
        Xk = self.X[:, start_index:end_index]

        # Update X
        for _ in range(10):
            Xk = tf.add(
                Xk,
                tf.multiply(
                    self.gamma_X,
                    tf.subtract(
                        tf.matmul(self.C, self.gradient(Yk, self.C, Xk), transpose_a=True),
                        tf.divide(Xk, self.sigma_X)
                    )
                )
            )
        self.X[:, start_index:end_index].assign(Xk)

        # Update C
        C_tilde = tf.add(
            self.C,
            tf.add(
                self.gradient_log_pi(Yk, self.C, Xk),
                tf.multiply(
                    tf.sqrt(2.0 * self.gamma_C),
                    tf.random.normal(tf.shape(self.C), mean=0.0, stddev=1.0, dtype=tf.float64)
                )
            )
            # self.gradient_log_pi(Yk, self.C, Xk)
        )
        
        # alpha = self.metropolis_hastings(Yk, Xk, C_tilde, self.C)
        # # print(alpha)
        # if tf.random.uniform([], dtype=tf.float64) < alpha:
        #     self.C.assign(C_tilde)
        self.C.assign(C_tilde)

        return tf.sigmoid(tf.matmul(self.C, Xk))