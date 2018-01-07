from __future__ import absolute_import, division


import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
import keras.backend as K


class TensorBoard(Callback):
    """Tensorboard basic visualizations"""

    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 write_graph=True,
                 write_images=False):
        super(TensorBoard, self).__init__()
        if K.backend() != 'tensorflow':
            raise RuntimeError('TensorBoard callback only works '
                               'with the TensorFlow backend.')
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_graph = write_graph
        self.write_images = write_images

    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()
        total_loss = self.model.total_loss
        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:
                for weight in layer.weights:
                    # dense_1/bias:0 > dense_1/bias_0
                    name = weight.name.replace(':', '_')
                    tf.summary.histogram(name, weight)
                    tf.summary.histogram(
                        '{}_gradients'.format(name),
                        K.gradients(total_loss, [weight])[0]
                    )
                    if self.write_images:
                        w_img = tf.squeeze(weight)
                        shape = w_img.get_shape()
                        if len(shape) > 1 and shape[0] > shape[1]:
                            w_img = tf.transpose(w_img)
                        if len(shape) == 1:
                            w_img = tf.expand_dims(w_img, 0)
                        w_img = tf.expand_dims(tf.expand_dims(w_img, 0), -1)
                        tf.summary.image(name, w_img)

                if hasattr(layer, 'output'):
                    tf.summary.histogram('{}_out'.format(layer.name),
                                         layer.output)
        self.merged = tf.summary.merge_all()

        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir,
                                                self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if self.validation_data and self.histogram_freq:
            if epoch % self.histogram_freq == 0:
                # TODO: implement batched calls to sess.run
                # (current call will likely go OOM on GPU)
                if self.model.uses_learning_phase:
                    cut_v_data = len(self.model.inputs)
                    val_data = self.validation_data[:cut_v_data][:32] + [0]
                    tensors = self.model.inputs + self.model.targets + [K.learning_phase()]
                else:
                    val_data = self.validation_data
                    tensors = self.model.inputs + self.model.targets

                feed_dict = dict(zip(tensors, val_data))
                sample_weights = self.model.sample_weights
                for w in sample_weights:
                    w_val = np.ones(len(val_data[0]), dtype=np.float32)
                    feed_dict.update({w.name: w_val})
                result = self.sess.run([self.merged], feed_dict=feed_dict)
                summary_str = result[0]
                self.writer.add_summary(summary_str, epoch)

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue

            if name[:3] != 'val':
                name = 'train_' + name

            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def on_train_end(self, _):
        self.writer.close()
