import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import argparse
import tensorflow as tf


def write_tfrecord(input_size, annotation_fp, image_dir, tfrecord_fp):
    """Landmark annotations in format x1 y1 x2 y2 ... x5 y5"""
    with open(annotation_fp, 'r') as f:
        annotations = [line.rstrip() for line in f.readlines()]

    examples = []
    for annotation in annotations:
        elements = annotation.split(' ')
        filename = elements[0]
        bbox = [int(_) for _ in elements[1:5]]
        landmarks = np.array([int(_) for _ in elements[5:]], dtype=np.float32)
        image = imread(os.path.join(image_dir, filename))

        image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        cropped_height, cropped_width = image.shape

        # Relocate landmarks to account for image cropping
        landmarks[::2] = landmarks[::2] - bbox[0]  # Subtract offset from x's
        landmarks[1::2] = landmarks[1::2] - bbox[1]  # Subtract offset from y's

        image = resize(image, (input_size, input_size))

        # Relocate landmarks to account for image resizing
        landmarks[::2] = landmarks[::2] / cropped_width * input_size
        landmarks[1::2] = landmarks[1::2] / cropped_height * input_size

        example = tf.train.Example(features=tf.train.Features(feature={
            'label_raw': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[landmarks.tostring()])),
            'image_raw': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[image.tostring()]))
        }))
        examples.append(example)

        with tf.python_io.TFRecordWriter(tfrecord_fp) as writer:
            for example in examples:
                writer.write(example.SerializeToString())


if __name__ == '__main__':
    def parse_arguments():
        parser = argparse.ArgumentParser()
        parser.add_argument('input_size', type=int,
                            help='The input size for specific net')
        parser.add_argument('annotation_fp', type=str)
        parser.add_argument('image_dir', type=str)
        parser.add_argument('output_tfrecord_fp', type=str)
        return parser.parse_args()


    args = parse_arguments()
    write_tfrecord(input_size=args.input_size,
                   annotation_fp=args.annotation_fp,
                   image_dir=args.image_dir,
                   tfrecord_fp=args.output_tfrecord_fp)
