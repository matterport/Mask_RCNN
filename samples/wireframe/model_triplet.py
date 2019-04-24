import numpy as np
import keras
import keras.backend as K
import keras.layers as KL
from mrcnn.triplet_loss import batch_all_triplet_loss
import tensorflow as tf
import samples.wireframe.database_actions as db_actions

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

    # Print
    #y_true = K.print_tensor(y_true, message='y_true is = ')

    # Run
    loss, _ = batch_all_triplet_loss(embeddings=y_pred, labels=y_true, margin=def_margin)
    return loss

def get_data(database):
    """
    :return: encodings array of (1024, n)
             labels list of (n)
    """
    query = "SELECT * FROM embeddings WHERE label IS NOT NULL"
    cursor, connection = db_actions.connect(database)
    cursor.execute(query)


    result_list = cursor.fetchall()
    encodings = np.zeros((1024, len(result_list)))
    labels = []

    for i in range(len(result_list)):
        encodings[:, i] = result_list[i][0]
        labels.append(result_list[i][1])
    encodings = np.nan_to_num(encodings)
    labels = [x for x in labels]
    return encodings.astype('float32'), labels
