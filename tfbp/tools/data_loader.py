import os, glob
import tensorflow as tf

class DataLoader():
    def __init__(self, input_path):
        self.input_path=input_path

    def distri_loader(self):
        pass


    def test_tfrecord(self):
        tf_path = glob.glob(os.path.join(self.input_path, '*.tfrec'))
        def from_tfrecord(serialized):
            features = tf.io.parse_single_example(
                serialized=serialized,
                features ={
                    'image': tf.io.VarLenFeature(tf.float32),
                    'image_shape': tf.io.VarLenFeature(tf.int64),
                    'label': tf.io.VarLenFeature(tf.int64),
                }
            )
            image_shape = tf.sparse.to_dense(features['image'])
            image = tf.reshape(tf.sparse.to_dense(features['image_shape']), (224,224))
            label = tf.sparse.to_dense(features['label'])
            return (image, label)

        dataset = tf.data.TFRecordDataset(
            filenames=tf_path,
            num_parallel_reads=tf.data.AUTOTUNE,
            compression_type='GZIP'
        ).map(from_tfrecord)
        return dataset