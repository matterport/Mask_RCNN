import keras.backend as K
import keras.layers as KL
from mrcnn.triplet_loss import batch_all_triplet_loss
import tensorflow as tf
from keras.models import Sequential

class Model:
    def __init__(self):
        self.model = self.build_model()
        self.compile_model()

    def build_model(self):
        base_model = KL.Input(shape=(None, 200, 128))
        x = base_model.output
        x = KL.TimeDistributed(KL.Dense(128), name='x')(x)
        full_model = KL.Activation('softmax')(x)
        return full_model

    def compile_model(self):
        self.model.compile(optimizer='adam', loss=triplet_loss,
                           metrics=['accuracy'])

def triplet_loss(y_true, y_pred):
    print("Y True (labels) shape: {}".format(K.int_shape(y_true)))
    print("Y Pred (embeddings) shape: {}".format(K.int_shape(y_pred)))

    # Reshape data
    y_pred = K.squeeze(y_pred, axis=1)
    y_pred = K.squeeze(y_pred, axis=1)
    y_true = tf.reshape(y_true, [-1])
    def_margin = tf.constant(1.0, dtype=tf.float32)

    # Print
    #y_true = K.print_tensor(y_true, message='y_true is = ')

    # Run
    loss, _ = batch_all_triplet_loss(embeddings=y_pred, labels=y_true, margin=def_margin)
    return loss