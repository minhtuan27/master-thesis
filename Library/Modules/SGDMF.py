import tensorflow as tf

class SGDMF(tf.Module):
    """
    Apply the SGDMF algorithm with Gaussian distribution model.
    It attempts to batch data by updating X in parallel and taking average update for C.
    
    Input:
        Y: tf.Tensor of shape (m, d), where m is the number of samples and d is the batch size
    Variables:
        C: tf.Tensor of shape (m, r)
    Output:
        X: tf.Tensor of shape (r, d)
    """

    def __init__(self, m, r, init_scale, alpha, beta, name=None):
        super().__init__(name=name)
        self.C = tf.Variable(tf.multiply(tf.random.truncated_normal([m, r], mean=1.0, stddev=0.02), init_scale))
        self.alpha = tf.constant(alpha)
        self.beta = tf.constant(beta)

    
    @tf.function
    def compute_xk(self, inputs):
        m = tf.reshape(inputs[0], [-1, 1])
        y = tf.reshape(inputs[1], [-1, 1])
        mC = tf.multiply(m, self.C)
        x = tf.matmul(tf.linalg.pinv(tf.matmul(mC, mC, transpose_a=True)), tf.matmul(mC, y, transpose_a=True))
        return tf.reshape(x, [-1])

    @tf.function
    def __call__(self, Y, k):
        k = tf.cast(k, tf.float32)

        # Define the mask
        M = tf.cast(tf.not_equal(Y, 0.0), tf.float32) # shape: (m, d)

        # Compute X
        X = tf.map_fn(
            fn=self.compute_xk, elems=(tf.transpose(M), tf.transpose(Y)), 
            fn_output_signature=tf.float32
        )
        X = tf.transpose(X) # shape: (r, d)

        # Update C and V
        self.C.assign(tf.add(
            self.C,
            tf.multiply(
                tf.matmul(
                    tf.multiply(M, tf.subtract(Y, tf.matmul(self.C, X))),
                    X, 
                    transpose_b=True
                ),
                tf.divide(self.alpha, tf.pow(k, self.beta))
            )
        )) # shape: (m, r)

        return X