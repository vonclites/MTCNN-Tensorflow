"""The code to test training process for rnet"""

# MIT License
#
# Copyright (c) 2017 Baoming Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import argparse

import tensorflow as tf
from mtcnn import train_net, RNet


def train_Rnet(training_data, base_lr, loss_weight,
               train_mode, num_epochs,
               load_model=False, load_filename=None,
               save_model=False, save_filename=None,
               num_iter_to_save=10000,
               device='/cpu:0', gpu_memory_fraction=1.0):

    pnet = tf.Graph()
    with pnet.as_default():
        with tf.device(device):
            train_net(Net=RNet,
                      training_data=training_data,
                      base_lr=base_lr,
                      loss_weight=loss_weight,
                      train_mode=train_mode,
                      num_epochs=num_epochs,
                      load_model=load_model,
                      load_filename=load_filename,
                      save_model=save_model,
                      save_filename=save_filename,
                      num_iter_to_save=num_iter_to_save,
                      gpu_memory_fraction=gpu_memory_fraction)


if __name__ == '__main__':
    def parse_arguments():
        parser = argparse.ArgumentParser()
        parser.add_argument('rnet_pretrained_weights_fp', type=str)
        parser.add_argument('output_model_fp', type=str)
        parser.add_argument('classifier_tfrecord_fp', type=str)
        parser.add_argument('localizer_tfrecord_fp', type=str)
        return parser.parse_args()


    args = parse_arguments()
    training_data = [args.classifier_tfrecord_fp,
                     args.localizer_tfrecord_fp]
    device = '/gpu:0'
    train_Rnet(training_data=training_data,
               base_lr=0.0001,
               loss_weight=[1.0, 0.5, 0.5],
               train_mode=3,
               num_epochs=[300, None, None],
               load_model=False,
               load_filename=args.rnet_pretrained_weights_fp,
               save_model=True,
               save_filename=args.output_model_fp,
               num_iter_to_save=5000,
               device=device,
               gpu_memory_fraction=1.0)
