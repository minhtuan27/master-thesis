import tensorflow as tf

class NMSE(tf.keras.metrics.Metric):
    def __init__(self, name='nmse', **kwargs):
        super(NMSE, self).__init__(name=name, **kwargs)
        self.total_squared_errors = self.add_weight(name='total_squared_errors', initializer='zeros')
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # If there is a mask, apply it
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            y_true *= sample_weight
            y_pred *= sample_weight

        # Update states
        squared_errors = tf.reduce_sum(tf.square(y_true - y_pred))
        samples = tf.reduce_sum(tf.square(y_true))
        self.total_squared_errors.assign_add(squared_errors)
        self.total_samples.assign_add(samples)

    def result(self):
        return self.total_squared_errors / self.total_samples

    def reset_states(self):
        self.total_squared_errors.assign(0.0)
        self.total_samples.assign(0.0)