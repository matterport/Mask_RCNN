from __future__ import absolute_import, division

from tensorflow.python import debug as tf_debug
import keras.backend as K


def keras_set_tf_debug():
    sess = K.get_session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    K.set_session(sess)
