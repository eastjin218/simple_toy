import os, glob
import tensorflow as tf

class DataLoader():
    def __init__(self, input_path, batch_size):
        self.input_path = input_path
        self.batch_size = batch_size

    def cls_tfrecord(self):
        tf_path = glob.glob(os.path.join(self.input_path, '*.tfrec'))
        def from_tfrecord(serialized):
            features = tf.io.parse_single_example(
                serialized=serialized,
                features ={
                    'image': tf.io.FixedLenFeature([],tf.string),
                    'image_shape_0': tf.io.FixedLenFeature([],tf.int64),
                    'image_shape_1': tf.io.FixedLenFeature([],tf.int64),
                    'image_shape_2': tf.io.FixedLenFeature([],tf.int64),
                    'label': tf.io.FixedLenFeature([],tf.int64),
                }
            )
            img_shape = (features['image_shape_0'], features['image_shape_1'])
            img = features['image']
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, img_shape) #224,224,3 / flaot32
            label = features['label']  # label(value = 3) / int64
            return img, label

        dataset = tf.data.TFRecordDataset(
            filenames=tf_path,
            num_parallel_reads=tf.data.AUTOTUNE,
        ).map(from_tfrecord)
        dataset= dataset.repeat(1)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset