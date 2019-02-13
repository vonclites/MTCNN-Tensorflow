import os
import numpy as np
import argparse
import tensorflow as tf
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.color import gray2rgb

from utils.annotation import annotate_image


def write_tfrecord(input_size,
                   annotation_fp,
                   image_dir,
                   tfrecord_fp,
                   landmark_padding,
                   jittered_samples_per_image=3):
    """Landmark annotations in format x1 y1 x2 y2 ... x5 y5"""
    with open(annotation_fp, 'r') as f:
        annotations = [line.rstrip() for line in f.readlines()]

    examples = []
    for annotation in annotations:
        elements = annotation.split(' ')
        filename = elements[0]
        bbox = [int(float(_)) for _ in elements[1:5]]
        landmarks = np.array([int(float(_)) for _ in elements[5:]],
                             dtype='float32')
        image = imread(os.path.join(image_dir, filename))
        if len(image.shape) == 2:
            image = gray2rgb(image)
        transformed_image, transformed_landmarks = transform_image_and_landmarks(
            image=image,
            landmarks=landmarks,
            bbox=bbox,
            target_size=input_size
        )
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[transformed_image.tostring()])),
            'label_raw': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[transformed_landmarks.tostring()]))
        }))
        examples.append(example)

        if jittered_samples_per_image > 0:
            _, positive_bboxes = generate_partial_and_positive_bboxes(
                image_size=image.shape[:2],
                bbox=bbox,
                landmarks=landmarks,
                landmark_padding=landmark_padding,
                positives_per_image=jittered_samples_per_image,
                partials_per_image=0,
                positive_threshold=0.65
            )
            for positive_bbox in positive_bboxes:
                transformed_image, transformed_landmarks = transform_image_and_landmarks(
                    image=image,
                    landmarks=landmarks,
                    bbox=positive_bbox,
                    target_size=input_size
                )
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image_raw': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[transformed_image.tostring()])),
                    'label_raw': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[transformed_landmarks.tostring()]))
                }))
                examples.append(example)

    with tf.python_io.TFRecordWriter(tfrecord_fp) as writer:
        for example in examples:
            writer.write(example.SerializeToString())


def transform_image_and_landmarks(image, landmarks, bbox, target_size):
    height, width, _ = image.shape
    cropped_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    cropped_height, cropped_width, _ = cropped_image.shape

    # Relocate landmarks to account for image cropping
    transformed_landmarks = np.ndarray(shape=landmarks.shape, dtype=np.float32)
    transformed_landmarks[::2] = landmarks[::2] - bbox[0]  # Subtract offset from x's
    transformed_landmarks[1::2] = landmarks[1::2] - bbox[1]  # Subtract offset from y's

    resized_image = resize(cropped_image,
                           (target_size, target_size),
                           preserve_range=True)
    resized_image = resized_image.astype('uint8')

    # Relocate landmarks to account for image resizing
    transformed_landmarks[::2] = transformed_landmarks[::2] / cropped_width * target_size
    transformed_landmarks[1::2] = transformed_landmarks[1::2] / cropped_height * target_size
    return resized_image, transformed_landmarks


def generate_partial_and_positive_bboxes(image_size,
                                         bbox,
                                         positives_per_image,
                                         partials_per_image,
                                         landmarks,
                                         landmark_padding,
                                         partial_threshold=0.6,
                                         positive_threshold=0.9,
                                         ):
    positives = []
    partials = []
    while (len(positives) < positives_per_image
           or len(partials) < partials_per_image):
        random_bbox = sample_distorted_bbox(image_size, bbox)
        if (landmarks[0] - random_bbox[0] < landmark_padding or
                landmarks[1] - random_bbox[1] < landmark_padding):
            continue
        if (iou(random_bbox, bbox) > positive_threshold
                and len(positives) < positives_per_image):
            positives.append(random_bbox)
        elif (iou(random_bbox, bbox) > partial_threshold
              and len(partials) < partials_per_image):
            partials.append(random_bbox)
    assert len(partials) == partials_per_image
    assert len(positives) == positives_per_image
    return partials, positives


def sample_distorted_bbox(image_size, bbox):
    height, width = image_size
    while True:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        size = np.random.randint(int(min(w, h) * 0.9),
                                 np.ceil(1.25 * max(w, h)))
        delta_x = np.random.randint(round(-w * 0.15), round(w * 0.15))
        delta_y = np.random.randint(round(-h * 0.15), round(h * 0.15))
        nx1 = int(max(bbox[0] + w / 2 + delta_x - size / 2, 0))
        ny1 = int(max(bbox[1] + h / 2 + delta_y - size / 2, 0))
        nx2 = nx1 + size
        ny2 = ny1 + size
        if nx2 < width and ny2 < height:
            return [nx1, ny1, nx2, ny2]


def iou(box1, box2):
    return intersection(box1, box2) / (union(box1, box2) + 0.00001)


def intersection(box1, box2):
    x1 = max(box1[0], box2[0])
    x2 = min(box1[2], box2[2])
    y1 = max(box1[1], box2[1])
    y2 = min(box1[3], box2[3])
    return area((x1, y1, x2, y2))


def union(box1, box2):
    return area(box1) + area(box2) - intersection(box1, box2)


def area(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


def check_tfrecords(tfrecord_fp, output_dir, input_shape):
    ds = tf.data.TFRecordDataset([tfrecord_fp])

    def map_fn(serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            {
                'label_raw': tf.FixedLenFeature([], tf.string),
                'image_raw': tf.FixedLenFeature([], tf.string)
            }
        )
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image.set_shape([input_shape * input_shape * 3])
        image = tf.reshape(image, [input_shape, input_shape, 3])
        label = tf.decode_raw(features['label_raw'], tf.float32)
        return image, label

    ds = ds.map(map_fn, num_parallel_calls=6)
    ds = ds.batch(64)
    ds_iter = ds.make_one_shot_iterator()
    images, labels = ds_iter.get_next()
    sess = tf.InteractiveSession()

    i = 0
    while True:
        try:
            xs, ys = sess.run([images, labels])
            for x, y in zip(xs, ys):
                annotate_image(x, y, has_bbox=False)
                imsave(os.path.join(output_dir, '{}.jpg'.format(i)), x)
                i += 1
        except tf.errors.OutOfRangeError:
            break


# write_tfrecord(input_size=12,
#                annotation_fp='/home/matt/Desktop/MTCNN_thermal_data/test_annotations.txt',
#                image_dir='/home/matt/Desktop/MTCNN_thermal_data/images',
#                tfrecord_fp='/home/matt/Desktop/check_jitter/narf.tfrecord',
#                landmark_padding=30,
#                jittered_samples_per_image=3)

# check_tfrecords(tfrecord_fp='/home/matt/Desktop/check_jitter/narf.tfrecord',
#                 output_dir='/home/matt/Desktop/check_jitter',
#                 input_shape=12)


# if __name__ == '__main__':
#     def parse_arguments():
#         parser = argparse.ArgumentParser()
#         parser.add_argument('input_size', type=int,
#                             help='The input size for specific net')
#         parser.add_argument('annotation_fp', type=str)
#         parser.add_argument('image_dir', type=str)
#         parser.add_argument('output_tfrecord_fp', type=str)
#         return parser.parse_args()
#
#
#     args = parse_arguments()
#     write_tfrecord(input_size=args.input_size,
#                    annotation_fp=args.annotation_fp,
#                    image_dir=args.image_dir,
#                    tfrecord_fp=args.output_tfrecord_fp)

