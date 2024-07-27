import tensorflow as tf

class CrossTab(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='cross_tab', **kwargs):
        super(CrossTab, self).__init__(name=name, **kwargs)
        self.confusion_matrix = self.add_weight(name='confusion_matrix', shape=(num_classes, num_classes), initializer='zeros')
        self.num_classes = num_classes
                                              
    def update_state(self, y_true, y_pred, sample_weight=None):
        # If there is a mask, apply it
        if sample_weight is not None:
            # Get non-zero indices of the mask
            mask_indices = tf.where(sample_weight > 0)

            # Gather non-masked values
            y_true = tf.gather_nd(y_true, mask_indices)
            y_pred = tf.gather_nd(y_pred, mask_indices)
        else:
            y_true = tf.reshape(y_true, [-1])
            y_pred = tf.reshape(y_pred, [-1])

        # Update confusion matrix
        self.confusion_matrix.assign_add(tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes))

    def result(self):
        return self.confusion_matrix
    
    def reset_states(self):
        self.confusion_matrix.assign(tf.zeros_like(self.confusion_matrix))