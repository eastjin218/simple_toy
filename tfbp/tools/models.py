import tensorflow as tf
from layers import ConvModule

class ClsModel(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(ClsModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.input_dim, self.input_dim, 3))
        # self.conv1 = tf.keras.layers.Conv2D(32, 3, strides=2, activation="relu")
        self.conv1 = ConvModule(32, 3, 2)
        self.max1  = tf.keras.layers.MaxPooling2D(3)
        self.bn1   = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(64, 3, activation="relu")
        self.bn2   = tf.keras.layers.BatchNormalization()
        self.drop  = tf.keras.layers.Dropout(0.3)

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.max1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop(x)
        return x

    def build_graph(self, raw_shape):
        x = tf.keras.layers.Input(shape=raw_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))