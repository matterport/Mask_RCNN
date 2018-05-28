import os
import argparse
import tensorflow as tf
from keras import backend as K
from tensorflow.python.framework import graph_util

import mrcnn.model as modellib
from mrcnn.config import Config

# Main arguments that can override the config file parameters
argparser = argparse.ArgumentParser(
    description='Export a model following a training')

argparser.add_argument('--model_path',
                       help='Locations of the h5 file containing the trained '
                            'weights. If none is provided, '
                            'the last trained h5 file fill be loaded within log_dirpath',
                       type=str, default= '')

argparser.add_argument('--log_dirpath', help='Location of the logdir containing the model training data',
                       type=str)

argparser.add_argument('--config_path',
                       help='Locations of the config file that was used to build the current model',
                       type=str)

def _main_(args):
    # Get the parameters from args
    config_path = args.config_path
    log_dirpath = args.log_dirpath
    model_path = args.model_path

    # Load the configs in the model folder
    config_inference = load_config_files(config_path=config_path)
    config_inference.display()

    # Create model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=config_inference,
                              model_dir=log_dirpath)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    if model_path == '':
        model_path = model.find_last()[1]

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    # All new operations will be in test mode from now on.
    K.set_learning_phase(0)

    # The filename is built from concantenating the decoders
    filename = os.path.splitext(os.path.basename(model_path))[0] + ".pb"

    # Get the TF session
    sess = K.get_session()

    # Get keras model and save
    model_keras = model.keras_model

    # Get the output heads name
    output_names_all = [output.name.split(':')[0] for output in model_keras.outputs]

    # Getthe graph to export
    graph_to_export = sess.graph

    # Freeze the variables in the graph and remove heads that were not selected
    # this will also cause the pb file to contain all the constant weights
    od_graph_def = graph_util.convert_variables_to_constants(sess,
                                                             graph_to_export.as_graph_def(),
                                                             output_names_all)

    model_dirpath = os.path.dirname(model_path)
    pb_filepath = os.path.join(model_dirpath, filename)
    print('Saving frozen graph {} ...'.format(os.path.basename(pb_filepath)))

    frozen_graph_path = pb_filepath
    with tf.gfile.GFile(frozen_graph_path, 'wb') as f:
        f.write(od_graph_def.SerializeToString())
    print('{} ops in the frozen graph.'.format(len(od_graph_def.node)))
    print()

def load_config_files(config_path):
    import importlib.machinery

    # Check if the file exists
    assert os.path.exists(config_path)
    # Dynamically load the config
    loader = importlib.machinery.SourceFileLoader('configs_', config_path)
    mod = loader.load_module()

    config = getattr(mod, Config.__name__)

    return config()

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
