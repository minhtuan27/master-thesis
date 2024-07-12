import tensorflow as tf

class BatchGaussianDF(tf.Module):
    """
    Apply a modified batch version of Dictionary Filter algorithm with Gaussian distribution model.
    
    Input:
        Y: tf.Tensor of shape (m, d), where m is the number of samples and d is the batch size
    Variables:
        C: tf.Tensor of shape (m, r)
        V: tf.Tensor of shape (r, r)
    Output:
        X: tf.Tensor of shape (r, d)
    """

    def __init__(self, m, r, init_scale, sigma, name=None):
        super().__init__(name=name)
        self.C = tf.Variable(tf.multiply(tf.random.truncated_normal([m, r], mean=1.0, stddev=0.02), init_scale))
        self.V = tf.Variable(tf.eye(r, dtype=tf.float32))
        self.sigmasq = sigma ** 2

    @tf.function
    def __call__(self, Y):
        # Define the mask
        M = tf.cast(tf.not_equal(Y, 0.0), tf.float32) # shape: (m, d)

        # Compute X
        @tf.function
        def compute_xk(inputs):
            m = tf.reshape(inputs[0], [-1, 1])
            y = tf.reshape(inputs[1], [-1, 1])
            mC = tf.multiply(m, self.C)
            x = tf.matmul(tf.linalg.pinv(tf.matmul(mC, mC, transpose_a=True)), tf.matmul(mC, y, transpose_a=True))
            return tf.reshape(x, [-1])

        X = tf.map_fn(
            fn=compute_xk, elems=(tf.transpose(M), tf.transpose(Y)), 
            fn_output_signature=tf.float32
        )
        X = tf.transpose(X) # shape: (r, d)

        # temp_cv = VX(\sigma^2I+X^TVX)^-1
        temp_cv = tf.matmul(
            tf.matmul(self.V, X),
            tf.linalg.pinv(
                tf.add(
                    tf.multiply(tf.eye(tf.shape(Y)[1]), self.sigmasq),
                    tf.matmul(X, tf.matmul(self.V, X), transpose_a=True)
                )
            )
        ) # shape: (d, d)

        # C = C + (M*(Y-CX))(temp_cv^T)
        self.C.assign_add(
            tf.matmul(
                tf.multiply(M, tf.subtract(Y, tf.matmul(self.C, X))),
                temp_cv,
                transpose_b=True
            )
        ) # shape: (m, r)

        # V = V - temp_cv X^TV
        self.V.assign_sub(
            tf.matmul(
                temp_cv,
                tf.matmul(X, self.V, transpose_a=True)
            )
        ) # shape: (r, r)

        return X