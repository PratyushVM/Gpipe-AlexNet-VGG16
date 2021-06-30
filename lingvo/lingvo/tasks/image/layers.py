# Defines classes GPipeLeNet5Pipeline and GPipeConvArch

from lingvo.core import base_layer
from lingvo.core import gpipe
from lingvo.core import layers
from lingvo.core import py_utils
import numpy as np
import lingvo.compat as tf


class GPipeLenet5PipeLine(gpipe.PipeliningLayer):
    @classmethod
    def Params(cls):
        p = super().Params()
        p.Define('filter_shapes', [(0, 0, 0, 0)],
                 'the filter shapes for the conv layers')
        p.Define('window_shapes', [(0, 0)],
                 'window shapes for the pool layers')
        p.Define('batch_size', 0, 'default batch size')
        p.Define('batch_norm', False, 'whether bn should be applied')
        p.Define('dropout_prob', 0.0, '')
        p.Define('number_micro_batches', 1, '')
        p.Define('splits', [0, 0, 0, 0], '')
        p.Define('softmax_input_dim', 0, '')
        p.Define('softmax_num_classes', 0, '')
        p.Define('conv1', layers.Conv2DLayer.Params(), '')
        p.Define('conv2', layers.Conv2DLayer.Params(), '')
        p.Define('pool1', layers.PoolingLayer.Params(), '')
        p.Define('pool2', layers.PoolingLayer.Params(), '')
        p.Define('fc1', layers.FCLayer.Params(), '')
        p.Define('fc2', layers.FCLayer.Params(), '')
        p.Define('sm1', layers.GPipeImageProcessingSoftmaxLayer.Params(), '')
        p.Define('input_shape', (0, 0, 0, 0), 'the shape of the input data')
        p.Define('class_weights', None, 'the default class weight')
        p.Define('class_ids', None, 'The default class labels')
        p.num_micro_batches = p.number_micro_batches
        p.batch_dim = p.batch_size
        return p

    def __init__(self, params):
        super().__init__(params)
        p = self.params
        p.conv1.Set(name='conv1', filter_shape=p.filter_shapes[0], filter_stride=(1, 1),
                    batch_norm=p.batch_norm)
        p.conv2.Set(name='conv2', filter_shape=p.filter_shapes[1], filter_stride=(1, 1),
                    batch_norm=p.batch_norm)
        p.pool1.Set(
            name='pool1', window_shape=p.window_shapes[0], window_stride=p.window_shapes[0])
        p.pool2.Set(
            name='pool2', window_shape=p.window_shapes[1], window_stride=p.window_shapes[0])
        shape = [tf.flags.FLAGS.minibatch_size] + list(p.input_shape)
        temp = layers.BaseConv2DLayer(p.conv1)
        shape = temp.OutShape(shape)
        temp = layers.PoolingLayer(p.pool1)
        shape = temp.OutShape(shape)
        temp = layers.BaseConv2DLayer(p.conv2)
        shape = temp.OutShape(shape)
        temp = layers.PoolingLayer(p.pool2)
        shape = temp.OutShape(shape)
        p.fc1.Set(name='fc1', input_dim=np.prod(shape[1:]),
                  output_dim=1000)
        p.fc2.Set(name='fc2', input_dim=1000, output_dim=p.softmax_input_dim)
        p.sm1.name = 'sm1'
        p.sm1.input_dim = p.softmax_input_dim
        p.sm1.num_classes = p.softmax_num_classes
        templars = []
        templars.append(p.conv1)
        templars.append(p.pool1)
        templars.append(p.conv2)
        templars.append(p.pool2)
        templars.append(p.fc1)
        templars.append(p.fc2)
        templars.append(p.sm1)
        cells = []
        cell_start = 0
        # To account for embedding layers in the pipeline.
        offset = 0
        for split, cell_end in enumerate(p.splits):
            # Layer 0 (embeddings) is always in split 0.
            sub = templars[cell_start:(cell_end + offset)]
            cell = gpipe.FeatureExtractionLayer.Params().Set(
                name='cell_{}'.format(split), sub=sub)
            cells.append(cell)
            cell_start = cell_end + offset
        p.cell_tpl = cells
        super().__init__(p)

    def FProp(self, theta, act, class_weights, class_ids):
        p = self.params
        logits = super().FProp(theta, act)  # padding is none
        reshaped_logits = tf.reshape(logits, [-1, p.sm1.num_classes])
        tgt_labels = tf.reshape(class_ids, [-1])
        num_splits = len(p.splits)
        smax = self.children['cell_{}'.format(num_splits - 1)].sm1
        smax_theta = theta['cell_{}'.format(num_splits - 1)].sm1
        per_example_xent, per_example_argmax = smax.XentLossFromLogits(
            smax_theta,
            reshaped_logits,
            class_weights=tf.reshape(class_weights, [-1]),
            class_ids=tgt_labels,
            class_probabilities=None)
        xent_shape = tf.shape(logits)[:2]
        label_weights = tf.reshape(
            tf.cast(class_weights, py_utils.FPropDtype(smax.params)), [-1])
        total_xent = tf.reduce_sum(per_example_xent * label_weights)
        total_weights = tf.reduce_sum(label_weights)
        return py_utils.NestedMap(
            logits=logits,
            log_probs=tf.nn.log_softmax(logits),
            per_example_argmax=per_example_argmax,
            per_example_xent=per_example_xent,
            per_example_weight=label_weights,
            total_xent=total_xent,
            total_weight=total_weights,
            avg_xent=total_xent / total_weights)


class GPipeConvArch(base_layer.BaseLayer):
    @classmethod
    def Params(cls):
        p = super().Params()
        p.Define('pipelinestack', GPipeLenet5PipeLine.Params(), '')
        return p

    @classmethod
    def CommonParams(cls, filter_shapes=[(0, 0, 0, 0)],
                     window_shapes=[(0, 0)],
                     batch_norm=False, dropout_prob=0.0,
                     softmax_input_dim=0,
                     softmax_num_classes=0,
                     batch_size=0,
                     number_micro_batches=0,
                     splits=[0, 0, 0, 0],
                     input_shape=(0, 0, 0, 0)):
        p = GPipeConvArch.Params()
        p.name = 'GPipeConvArch'
        p.pipelinestack.filter_shapes = filter_shapes
        p.pipelinestack.window_shapes = window_shapes
        p.pipelinestack.batch_norm = batch_norm
        p.pipelinestack.dropout_prob = dropout_prob
        p.pipelinestack.softmax_input_dim = softmax_input_dim
        p.pipelinestack.softmax_num_classes = softmax_num_classes
        p.pipelinestack.batch_size = batch_size
        p.pipelinestack.number_micro_batches = number_micro_batches
        p.pipelinestack.splits = splits
        p.pipelinestack.input_shape = input_shape
        p.pipelinestack.num_micro_batches = number_micro_batches
        return p

    def __init__(self, params):
        super().__init__(params)
        p = self.params
        self.CreateChild('pipelinestack', p.pipelinestack)

    def FProp(self, theta, act, class_weights, class_ids):
        return self.pipelinestack.FProp(theta.pipelinestack, act, class_weights, class_ids)
