import argparse
import math
import os, glob, io
from typing import Tuple
from collections import defaultdict
import numpy as np
import tensorflow as tf
import tqdm
from PIL import Image

RESOLUTION = 256

def load_cls_dataset(args):
    img_path = glob.glob(os.path.join(args.input_path, '**/*.jpg'),recursive=True)
    train_ds=defaultdict(list)
    test_ds =defaultdict(list)
    for idx, i in enumerate(img_path[:100]):
        if idx < len(img_path)*0.8:
            train_ds['image'].append(Image.open(i))
            train_ds['label'].append(int(i.split('/')[-2]))
        else:
            train_ds['image'].append(Image.open(i))
            train_ds['label'].append(int(i.split('/')[-2]))
    return train_ds, test_ds


def process_image(img, resize):
    img = img.resize((resize, resize), Image.LANCZOS)
    return img

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_tfrecord(image: Image, label: Image, resize: int):
    img = process_image(image,resize)
    img_dims = np.array(img).shape
   
    byte_io = io.BytesIO()
    img.save(byte_io, format='PNG')
    byte_img = byte_io.getvalue()

    return tf.train.Example(
        features=tf.train.Features(
            feature={
                "image": _bytes_feature(byte_img),
                "image_shape_0": _int64_feature(img_dims[0]),
                "image_shape_1": _int64_feature(img_dims[1]),
                "image_shape_2": _int64_feature(img_dims[2]),
                "label": _int64_feature(int(label)),
            }
        )
    ).SerializeToString()


def write_tfrecords(root_dir, dataset, split, batch_size, resize):
    print(f"Preparing TFRecords for split: {split}.")

    for step in tqdm.tnrange(int(math.ceil(len(dataset['image']) / batch_size))):
        temp_img_ds = dataset['image'][step * batch_size : (step + 1) * batch_size]
        temp_label_ds = dataset['label'][step * batch_size : (step + 1) * batch_size]
        shard_size = len(temp_img_ds)
        filename = os.path.join(
            root_dir, "{}-{:02d}-{}.tfrec".format(split, step, shard_size)
        )

        with tf.io.TFRecordWriter(filename) as out_file:
            for i in range(shard_size):
                image = temp_img_ds[i]
                label = temp_label_ds[i]
                example = create_tfrecord(image, label, resize)
                out_file.write(example)
            print("Wrote file {} containing {} records".format(filename, shard_size))


def main(args):
    train_ds, val_ds = load_cls_dataset(args)
    print("Dataset loaded from HF.")

    if not os.path.exists(args.root_tfrecord_dir):
        os.makedirs(args.root_tfrecord_dir, exist_ok=True)

    print(args.resize)

    write_tfrecords(
        args.root_tfrecord_dir, train_ds, "train", args.batch_size, args.resize
    )
    write_tfrecords(args.root_tfrecord_dir, val_ds, "val", args.batch_size, args.resize)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", help="Train and test split.", default=0.2, type=float
    )
    parser.add_argument(
        "--seed",
        help="Seed to be used while performing train-test splits.",
        default=2022,
        type=int,
    )
    parser.add_argument(
        "--input_path",
        help="Seed to be used while performing train-test splits.",
        default='/home/simple_toy/tfbp/cls_data_test/',
        type=str,
    )
    parser.add_argument(
        '-o',
        "--root_tfrecord_dir",
        help="Root directory where the TFRecord shards will be serialized.",
        default="/home/simple_toy/tfbp/cls-tfrecords",
        type=str,
    )
    parser.add_argument(
        "--batch_size",
        help="Number of samples to process in a batch before serializing a single TFRecord shard.",
        default=32,
        type=int,
    )
    parser.add_argument(
        "--resize",
        help="Width and height size the image will be resized to. No resizing will be applied when this isn't set.",
        type=int,
        default=224,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)