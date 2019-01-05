
import tensorflow as tf
from keras.callbacks import Callback

import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM


class cucu_summaryCallback(Callback):

    def __init__(self, log_dir='./logs',
                histogram_freq=0,
                batch_size=32,
                write_graph=True,
                write_grads=False,
                write_images=False):
        super(cucu_summaryCallback, self).__init__()
        if K.backend() != 'tensorflow':
            raise RuntimeError('TensorBoard callback only works '
                            'with the TensorFlow backend.')
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.mergedTensorboardSumm = None
        self.write_graph = write_graph
        self.write_grads = write_grads
        self.write_images = write_images
        self.batch_size = batch_size

    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()
        with tf.name_scope('performance'):
            if self.histogram_freq and self.mergedTensorboardSumm is None:
                for layer in self.model.layers:

                    for weight in layer.weights:
                        mapped_weight_name = weight.name.replace(':', '_')
                        tf.summary.histogram(mapped_weight_name, weight)
                        if self.write_grads:
                            grads = model.optimizer.get_gradients(model.total_loss,
                                                                weight)

                            def is_indexed_slices(grad):
                                return type(grad).__name__ == 'IndexedSlices'
                            grads = [
                                grad.values if is_indexed_slices(grad) else grad
                                for grad in grads]
                            tf.summary.histogram('{}_grad'.format(mapped_weight_name), grads)
                        if self.write_images:
                            w_img = tf.squeeze(weight)
                            shape = K.int_shape(w_img)
                            if len(shape) == 2:  # dense layer kernel case
                                if shape[0] > shape[1]:
                                    w_img = tf.transpose(w_img)
                                    shape = K.int_shape(w_img)
                                w_img = tf.reshape(w_img, [1,
                                                        shape[0],
                                                        shape[1],
                                                        1])
                            elif len(shape) == 3:  # convnet case
                                if K.image_data_format() == 'channels_last':
                                    # switch to channels_first to display
                                    # every kernel as a separate image
                                    w_img = tf.transpose(w_img, perm=[2, 0, 1])
                                    shape = K.int_shape(w_img)
                                w_img = tf.reshape(w_img, [shape[0],
                                                        shape[1],
                                                        shape[2],
                                                        1])
                            elif len(shape) == 1:  # bias case
                                w_img = tf.reshape(w_img, [1,
                                                        shape[0],
                                                        1,
                                                        1])
                            else:
                                # not possible to handle 3D convnets etc.
                                continue

                            shape = K.int_shape(w_img)
                            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
                            tf.summary.image(mapped_weight_name, w_img)

                    # if hasattr(layer, 'output'):
                    #     if(layer.output.dtype == bool):
                    #         continue
                    #     tf.summary.histogram('{}_out'.format(layer.name),
                    #                         layer.output)
            self.mergedTensorboardSumm = tf.summary.merge_all()

            if self.write_graph:
                self.writer = tf.summary.FileWriter(self.log_dir,
                                                    self.sess.graph)
            else:
                self.writer = tf.summary.FileWriter(self.log_dir)
    def on_batch_end(self, batch, logs=None):

        self.validation_data = logs
        del self.validation_data["batch"]
        del self.validation_data["size"]
        result = self.sess.run([self.mergedTensorboardSumm])
        summary_str = result[0]
        self.writer.add_summary(summary_str, batch)
        

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # if not self.validation_data and self.histogram_freq:
        #     raise ValueError('If printing histograms, validation_data must be '
        #                     'provided, and cannot be a generator.')
        # if self.validation_data and self.histogram_freq:
        #     if epoch % self.histogram_freq == 0:

        #         val_data = self.validation_data
        #         tensors = (self.model.inputs +
        #                 self.model.targets +
        #                 self.model.sample_weights)

        #         if self.model.uses_learning_phase:
        #             tensors += [K.learning_phase()]

        #         assert len(val_data) == len(tensors)
        #         val_size = val_data[0].shape[0]
        #         i = 0
        #         while i < val_size:
        #             step = min(self.batch_size, val_size - i)
        #             if self.model.uses_learning_phase:
        #                 # do not slice the learning phase
        #                 batch_val = [x[i:i + step] for x in val_data[:-1]]
        #                 batch_val.append(val_data[-1])
        #             else:
        #                 batch_val = [x[i:i + step] for x in val_data]
        #             assert len(batch_val) == len(tensors)
        #             feed_dict = dict(zip(tensors, batch_val))
        #             result = self.sess.run([self.mergedTensorboardSumm], feed_dict=feed_dict)
        #             summary_str = result[0]
        #             self.writer.add_summary(summary_str, epoch)
        #             i += self.batch_size
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def on_train_end(self, _):
        self.writer.close()

