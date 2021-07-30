# Lint as: python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train models on MNIST data."""

from lingvo import model_registry
from lingvo.core import base_model_params
from lingvo.tasks.image import classifier
from lingvo.tasks.image import input_generator
import lingvo.compat as tf
from lingvo.tasks.image import layers as convarch_layers


class Base(base_model_params.SingleTaskModelParams):
    """Input params for MNIST."""

    @property
    def path(self):
        # Generated using lingvo/tools:keras2ckpt.
        return '/tmp/mnist/mnist'

    def Train(self):
        p = input_generator.MnistTrainInput.Params()
        p.ckpt = self.path
        return p

    def Test(self):
        p = input_generator.MnistTestInput.Params()
        p.ckpt = self.path
        return p

    def Dev(self):
        return self.Test()


@model_registry.RegisterSingleTaskModel
class LeNet5(Base):
    """LeNet params for MNIST classification."""
    BN = True
    DROP = 0.0

    def Task(self):
        p = classifier.ModelV1.Params()
        p.name = 'lenet5'
        # Overall architecture:
        #   conv, maxpool, conv, maxpool, fc, fc, softmax
        p.filter_shapes = [(5, 5, 1, 6), (5, 5, 6, 16)]
        p.window_shapes = [(2, 2), (2, 2)]
        p.batch_norm = self.BN
        p.dropout_prob = self.DROP
        p.softmax.input_dim = 84
        p.softmax.num_classes = 10
        p.train.save_interval_seconds = 10  # More frequent checkpoints.
        p.eval.samples_per_summary = 0  # Eval the whole set.
        p.train.max_steps = 5 * (60255//256)  # 5 epochs

        return p


@model_registry.RegisterSingleTaskModel
class GPipeLeNet5(Base):

    BN = False
    DROP = 0.0
    BATCH_SIZE = tf.flags.FLAGS.minibatch_size
    # GPUS = 1
    SPLITS = [7]  # [2 * (i + 1) for i in range(GPUS)]
    LAYERS = SPLITS[-1]
    NUM_MICRO_BATCHES = 8

    def Task(self):
        p = classifier.GPipeModel.Params()
        p.name = 'gpipelenet5'
        p.convarch = convarch_layers.GPipeConvArch.CommonParams(filter_shapes=[(5, 5, 1, 6), (5, 5, 6, 16)],
                                                                window_shapes=[
                                                                    (2, 2), (2, 2)],
                                                                batch_norm=self.BN,
                                                                dropout_prob=self.DROP,
                                                                softmax_input_dim=84,
                                                                softmax_num_classes=10,
                                                                batch_size=self.BATCH_SIZE,
                                                                number_micro_batches=self.NUM_MICRO_BATCHES,
                                                                splits=self.SPLITS)
        p.train.save_interval_seconds = 10  # More frequent checkpoints.
        p.eval.samples_per_summary = 0  # Eval the whole set.
        p.softmax.input_dim = 84
        p.softmax.num_classes = 10
        p.train.max_steps = 5 * (60255//256)  # 5 epochs
        print('finished getting params')
        return p


@model_registry.RegisterSingleTaskModel
class VGG16(Base):

    BN = False
    DROP = 0.0

    def Task(self):
        p = classifier.ModelVGG16.Params()
        p.name = 'vgg16'

        p.filter_shapes = [(3, 3, 1, 64), (3, 3, 64, 64), (3, 3, 64, 128), (3, 3, 128, 128), (3, 3, 128, 256), (3, 3, 256, 256), (
            3, 3, 256, 256), (3, 3, 256, 512), (3, 3, 512, 512), (3, 3, 512, 512), (3, 3, 512, 512), (3, 3, 512, 512), (3, 3, 512, 512)]
        p.window_shapes = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]
        p.batch_norm = self.BN
        p.dropout_prob = self.DROP
        p.softmax.input_dim = 4096
        p.softmax.num_classes = 10
        p.train.save_interval_seconds = 10  # More frequent checkpoints.
        p.eval.samples_per_summary = 0  # Eval the whole set.
        p.train.max_steps = 5 * (60255//256)  # 5 epochs

        return p


@model_registry.RegisterSingleTaskModel
class GPipeVGG16(Base):

    BN = False
    DROP = 0.0
    BATCH_SIZE = tf.flags.FLAGS.minibatch_size
    # GPUS = 1
    SPLITS = [21]  # [2 * (i + 1) for i in range(GPUS)]
    LAYERS = SPLITS[-1]
    NUM_MICRO_BATCHES = 8

    def Task(self):
        p = classifier.GPipeModelVGG16.Params()
        p.name = 'gpipevgg16'
        p.convarch = convarch_layers.GPipeConvArchVGG16.CommonParams(filter_shapes=[(3, 3, 1, 64), (3, 3, 64, 64), (3, 3, 64, 128), (3, 3, 128, 128), (3, 3, 128, 256), (3, 3, 256, 256), (
            3, 3, 256, 256), (3, 3, 256, 512), (3, 3, 512, 512), (3, 3, 512, 512), (3, 3, 512, 512), (3, 3, 512, 512), (3, 3, 512, 512)],
            window_shapes=[
            (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)],
            batch_norm=self.BN,
            dropout_prob=self.DROP,
            softmax_input_dim=4096,
            softmax_num_classes=10,
            batch_size=self.BATCH_SIZE,
            number_micro_batches=self.NUM_MICRO_BATCHES,
            splits=self.SPLITS)
        p.train.save_interval_seconds = 10  # More frequent checkpoints.
        p.eval.samples_per_summary = 0  # Eval the whole set.
        p.softmax.input_dim = 4096
        p.softmax.num_classes = 10
        p.train.max_steps = 5 * (60255//256)  # 5 epochs
        print('finished getting params')
        return p


@model_registry.RegisterSingleTaskModel
class AlexNet(Base):

    BN = False
    DROP = 0.0

    def Task(self):
        p = classifier.ModelAlexNet.Params()
        p.name = 'alexnet'

        p.filter_shapes = [(11, 11, 1, 96), (5, 5, 96, 256),
                           (3, 3, 256, 384), (3, 3, 384, 384), (3, 3, 384, 256)]
        p.window_shapes = [(2, 2), (2, 2), (2, 2)]
        p.batch_norm = self.BN
        p.dropout_prob = self.DROP
        p.softmax.input_dim = 4096
        p.softmax.num_classes = 10
        p.train.save_interval_seconds = 10  # More frequent checkpoints.
        p.eval.samples_per_summary = 0  # Eval the whole set.
        p.train.max_steps = 5  # * (60255//256)  # 5 epochs

        return p


@model_registry.RegisterSingleTaskModel
class GPipeAlexNet(Base):

    BN = False
    DROP = 0.0
    BATCH_SIZE = tf.flags.FLAGS.minibatch_size
    # GPUS = 1
    SPLITS = [11]  # [2 * (i + 1) for i in range(GPUS)]
    LAYERS = SPLITS[-1]
    NUM_MICRO_BATCHES = 8

    def Task(self):
        p = classifier.GPipeModelVGG16.Params()
        p.name = 'gpipealexnet'
        p.convarch = convarch_layers.GPipeConvArchAlexNet.CommonParams(filter_shapes=[(11, 11, 1, 96), (5, 5, 96, 256),
                                                                                      (3, 3, 256, 384), (3, 3, 384, 384), (3, 3, 384, 256)],
                                                                       window_shapes=[
            (2, 2), (2, 2), (2, 2)],
            batch_norm=self.BN,
            dropout_prob=self.DROP,
            softmax_input_dim=4096,
            softmax_num_classes=10,
            batch_size=self.BATCH_SIZE,
            number_micro_batches=self.NUM_MICRO_BATCHES,
            splits=self.SPLITS)
        p.train.save_interval_seconds = 10  # More frequent checkpoints.
        p.eval.samples_per_summary = 0  # Eval the whole set.
        p.softmax.input_dim = 4096
        p.softmax.num_classes = 10
        p.train.max_steps = 5 * (60255//256)  # 5 epochs
        print('finished getting params')
        return p


""" Note : GPipe currently does not support BN for our experimental convlayers 
    Whereas Lingvo supports BN and gives moving mean, moving variance,etc -
    Use BN = True only in the case of Non-GPipe """
