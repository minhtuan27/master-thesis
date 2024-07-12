import tensorflow as tf

class NMSE(tf.keras.metrics.Metric):
    def __init__(self, name='nmse', **kwargs):
        super(NMSE, self).__init__(name=name, **kwargs)
        self.total_mse = self.add_weight(name='total_mse', initializer='zeros')
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # If there is a mask, apply it
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            y_true *= sample_weight
            y_pred *= sample_weight

        # Update states
        mse = tf.reduce_sum(tf.square(y_true - y_pred))
        samples = tf.reduce_sum(tf.square(y_true))
        self.total_mse.assign_add(mse)
        self.total_samples.assign_add(samples)

    def result(self):
        return self.total_mse / self.total_samples

    def reset_states(self):
        self.total_mse.assign(0.0)
        self.total_samples.assign(0.0)