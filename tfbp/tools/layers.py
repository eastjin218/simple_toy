import tensorflow as tf

class ConvModule(tf.keras.layers.Layer):
    def __init__(self, filter, kernel_size, strides=1, padding='same'):
        super(ConvModule, self).__init__()
        self.conv = tf.keras.layers.Conv2D(
            filters= filter,
            kernel_size=kernel_size,
            strides = (strides, strides),
            padding=padding
        )
        self.bn  = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x