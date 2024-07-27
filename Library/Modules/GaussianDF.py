import tensorflow as tf

class GaussianDF(tf.Module):
    """
    Apply the Dictionary Filter algorithm with Gaussian distribution model.
    It attempts to batch data by updating X in parallel and taking average update for C and V.
    
    Input:
        Y: tf.Tensor of shape (m, d), where m is the number of samples and d is the batch size
    Variables:
        C: tf.Tensor of shape (m, r)
        V: tf.Tensor of shape (r, r)
    Output:
        X: tf.Tensor of shape (r, d)
    """

    def __init__(self, m, r, init_scale, train_lambda, name=None):
        super().__init__(name=name)
        self.C = tf.Variable(tf.multiply(tf.random.truncated_normal([m, r], mean=1.0, stddev=0.02), init_scale))
        self.V = tf.Variable(tf.eye(r, dtype=tf.float32))
        self.train_lambda = tf.constant(train_lambda)

    
    @tf.function
    def compute_xk(self, inputs):
        m = tf.reshape(inputs[0], [-1, 1])
        y = tf.reshape(inputs[1], [-1, 1])
        mC = tf.multiply(m, self.C)
        x = tf.matmul(tf.linalg.pinv(tf.matmul(mC, mC, transpose_a=True)), tf.matmul(mC, y, transpose_a=True))
        return tf.reshape(x, [-1])

    @tf.function
    def __call__(self, Y, return_V_updates=False):
        # Define the mask
        M = tf.cast(tf.not_equal(Y, 0.0), tf.float32) # shape: (m, d)

        # Compute X
        X = tf.map_fn(
            fn=self.compute_xk, elems=(tf.transpose(M), tf.transpose(Y)), 
            fn_output_signature=tf.float32
        )
        X = tf.transpose(X) # shape: (r, d)

        # Update C and V
        denominator = tf.multiply(
            tf.add(
                tf.reduce_sum(tf.multiply(X, tf.matmul(self.V, X)), axis=0, keepdims=True),
                self.train_lambda
            ),
            tf.cast(tf.reshape(tf.shape(Y)[1], [1, 1]), tf.float32)
        ) # shape: (1, d)

        reduced_X = tf.divide(X, denominator) # shape: (r, d)

        temp = tf.matmul(reduced_X, self.V, transpose_a=True) # shape: (d, r)

        self.C.assign(tf.add(
            self.C,
            tf.matmul(
                tf.multiply(M, tf.subtract(Y, tf.matmul(self.C, X))),
                temp
            )
        )) # shape: (m, r)

        prev_V = self.V

        self.V.assign(tf.subtract(
            self.V,
            tf.matmul(
                tf.matmul(self.V, X),
                temp
            )
        )) # shape: (r, r)

        if return_V_updates:
            prev_V = tf.repeat(tf.reshape(prev_V, [1, -1]), tf.shape(denominator)[1], axis=0)
            V_updates = tf.divide(prev_V, tf.repeat(tf.reshape(denominator, [-1, 1]), tf.shape(prev_V)[1], axis=1))
            return X, V_updates

        return X