import os
import re
from prepare_data import gen_shuffle_data
from prepare_data import gen_tfdata_12net
from prepare_data import gen_tfdata_24net
from prepare_data import gen_tfdata_48net
from prepare_data import gen_tfdata_landmarks
from prepare_data import tf_gen_12net_hard_example
from prepare_data import tf_gen_24net_hard_example
from src import mtcnn_pnet_test
from src import mtcnn_rnet_test
from src import mtcnn_onet_test


def get_latest_ckpt(model_dir, ckpt_prefix):
    ckpts = os.listdir(model_dir)
    pattern = '{}-\d+.meta\Z'.format(ckpt_prefix)
    r = re.compile(pattern)
    m = list(filter(r.match, ckpts))
    m.sort()
    return os.path.join(model_dir, m[-1][:-5])


ANNO_FP = '/home/matt/Desktop/MTCNN_thermal_16bit/train_annotations.txt'
IMAGE_DIR = '/home/matt/Desktop/MTCNN_thermal_16bit/images'
ROOT_DIR = '/home/matt/Desktop/MTCNN_thermal_16bit'
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
PNET_DIR = os.path.join(MODEL_DIR, 'pnet')
RNET_DIR = os.path.join(MODEL_DIR, 'rnet')
ONET_DIR = os.path.join(MODEL_DIR, 'onet')


gen_shuffle_data.main(
    input_size=12,
    annotation_fp=ANNO_FP,
    image_dir=IMAGE_DIR,
    output_dir=ROOT_DIR
)
gen_tfdata_12net.main(
    input_size=12,
    classifier_tfrecord_fp=os.path.join(ROOT_DIR, 'pnet_cls.tfrecord'),
    localizer_tfrecord_fp=os.path.join(ROOT_DIR, 'pnet_bbox.tfrecord'),
    root_data_dir=ROOT_DIR
)
gen_tfdata_landmarks.write_tfrecord(
    input_size=12,
    annotation_fp=ANNO_FP,
    image_dir=IMAGE_DIR,
    tfrecord_fp=os.path.join(ROOT_DIR, 'pnet_landmarks.tfrecord')
)

mtcnn_pnet_test.train_Pnet(
    training_data=[os.path.join(ROOT_DIR, 'pnet_cls.tfrecord'),
                   os.path.join(ROOT_DIR, 'pnet_bbox.tfrecord'),
                   os.path.join(ROOT_DIR, 'pnet_landmarks.tfrecord')],
    base_lr=0.0001,
    loss_weight=[1.0, 0.5, 0.5],
    train_mode=3,
    num_epochs=[1000, 1000, 1000],
    load_model=False,
    load_filename='/home/matt/dev/MTCNN-Tensorflow/'
                  'pretrained/initial_weight_pnet.npy',
    save_model=True,
    save_filename=os.path.join(PNET_DIR, 'pnet'),
    num_iter_to_save=500,
    device='/gpu:0',
    gpu_memory_fraction=0.9
)
tf_gen_12net_hard_example.main(
    annotation_fp=ANNO_FP,
    image_dir=IMAGE_DIR,
    model_fp=get_latest_ckpt(model_dir=PNET_DIR, ckpt_prefix='pnet'),
    output_dir=ROOT_DIR
)

gen_shuffle_data.main(
    input_size=24,
    annotation_fp=ANNO_FP,
    image_dir=IMAGE_DIR,
    output_dir=ROOT_DIR
)
gen_tfdata_24net.main(
    input_size=24,
    classifier_tfrecord_fp=os.path.join(ROOT_DIR, 'rnet_cls.tfrecord'),
    localizer_tfrecord_fp=os.path.join(ROOT_DIR, 'rnet_bbox.tfrecord'),
    root_data_dir=ROOT_DIR
)
gen_tfdata_landmarks.write_tfrecord(
    input_size=24,
    annotation_fp=ANNO_FP,
    image_dir=IMAGE_DIR,
    tfrecord_fp=os.path.join(ROOT_DIR, 'rnet_landmarks.tfrecord')
)

mtcnn_rnet_test.train_Rnet(
    training_data=[os.path.join(ROOT_DIR, 'rnet_cls.tfrecord'),
                   os.path.join(ROOT_DIR, 'rnet_bbox.tfrecord'),
                   os.path.join(ROOT_DIR, 'rnet_landmarks.tfrecord')],
    base_lr=0.0001,
    loss_weight=[1.0, 0.5, 0.5],
    train_mode=3,
    num_epochs=[2000, 2000, 2000],
    load_model=False,
    load_filename='/home/matt/dev/MTCNN-Tensorflow/'
                  'pretrained/initial_weight_rnet.npy',
    save_model=True,
    save_filename=os.path.join(RNET_DIR, 'rnet'),
    num_iter_to_save=500,
    device='/gpu:0',
    gpu_memory_fraction=0.9
)
tf_gen_24net_hard_example.main(
    annotation_fp=ANNO_FP,
    image_dir=IMAGE_DIR,
    pnet_model_fp=get_latest_ckpt(model_dir=PNET_DIR, ckpt_prefix='pnet'),
    rnet_model_fp=get_latest_ckpt(model_dir=RNET_DIR, ckpt_prefix='rnet'),
    output_dir=ROOT_DIR
)
gen_shuffle_data.main(
    input_size=48,
    annotation_fp=ANNO_FP,
    image_dir=IMAGE_DIR,
    output_dir=ROOT_DIR
)
gen_tfdata_48net.main(
    input_size=48,
    classifier_tfrecord_fp=os.path.join(ROOT_DIR, 'onet_cls.tfrecord'),
    localizer_tfrecord_fp=os.path.join(ROOT_DIR, 'onet_bbox.tfrecord'),
    root_data_dir=ROOT_DIR
)
gen_tfdata_landmarks.write_tfrecord(
    input_size=48,
    annotation_fp=ANNO_FP,
    image_dir=IMAGE_DIR,
    tfrecord_fp=os.path.join(ROOT_DIR, 'onet_landmarks.tfrecord')
)

mtcnn_onet_test.train_Onet(
    training_data=[os.path.join(ROOT_DIR, 'onet_cls.tfrecord'),
                   os.path.join(ROOT_DIR, 'onet_bbox.tfrecord'),
                   os.path.join(ROOT_DIR, 'onet_landmarks.tfrecord')],
    base_lr=0.0005,
    loss_weight=[1.0, 0.5, 1.0],
    train_mode=3,
    num_epochs=[5000, 5000, 5000],
    load_model=False,
    load_filename='/home/matt/dev/MTCNN-Tensorflow/'
                  'pretrained/initial_weight_onet.npy',
    save_model=True,
    save_filename=os.path.join(ONET_DIR, 'onet'),
    num_iter_to_save=500,
    device='/gpu:0',
    gpu_memory_fraction=0.9
)
