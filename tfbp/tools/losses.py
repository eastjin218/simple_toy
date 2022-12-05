import tensorflow as tf

class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, threshold=1, strategy=None):
        super().__init__()
        self.threshold= threshold

    def call(self, y_true, y_pred):
        if strategy:
            with strategy.scope():
                loss_obj = tf.kera
        error = y_true - t_pred
