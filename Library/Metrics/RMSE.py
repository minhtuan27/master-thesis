import tensorflow as tf

class RMSE(tf.keras.metrics.Metric):
    def __init__(self, name='rmse', **kwargs):
        super(RMSE, self).__init__(name=name, **kwargs)
        self.total_squared_errors = self.add_weight(name='total_squared_errors', initializer='zeros')
        self.total_count = self.add_weight(name='total_count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # If there is a mask, apply it
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            y_true *= sample_weight
            y_pred *= sample_weight

        # Update states
        squared_errors = tf.reduce_sum(tf.square(y_true - y_pred))
        count = tf.math.count_nonzero(y_true, dtype=self.dtype)
        self.total_squared_errors.assign_add(squared_errors)
        self.total_count.assign_add(count)

    def result(self):
        return tf.sqrt(self.total_squared_errors / self.total_count)

    def reset_states(self):
        self.total_squared_errors.assign(0.0)
        self.total_count.assign(0.0)