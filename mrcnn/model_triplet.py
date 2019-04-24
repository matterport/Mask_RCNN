import numpy as np
import keras
import keras.backend as K
import keras.layers as KL
from mrcnn.triplet_loss import batch_all_triplet_loss
import tensorflow as tf

class Model:
    def __init__(self):
        self.model = self.build_model()
        self.compile_model()

    def build_model(self):
        base = KL.Input(shape=(1024,))
        x = KL.Dense(128)(base)
        x = KL.Activation('softmax')(x)
        full_model = keras.Model(inputs=base, outputs=x)
        return full_model

    def compile_model(self):
        self.model.compile(optimizer='adam', loss=triplet_loss,
                           metrics=['accuracy'])

def triplet_loss(y_true, y_pred):
    print("Y True (labels) shape: {}".format(K.int_shape(y_true)))
    print("Y Pred (embeddings) shape: {}".format(K.int_shape(y_pred)))

    # Reshape data
    y_true = tf.reshape(y_true, [-1])
    def_margin = tf.constant(1.0, dtype=tf.float32)

    # Run
    loss, _ = batch_all_triplet_loss(embeddings=y_pred, labels=y_true, margin=def_margin)
    return loss


embeddings = np.ones((2, 1024))
labels = np.ones(2)
model = Model()
model.model.fit(embeddings, labels, batch_size=10, epochs=1)