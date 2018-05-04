# Mobile Mask R-CNN
This is a Mask R-CNN implementation with MobileNet V1/V2 as Backbone architecture to be finally able to deploy it on mobile devices such as the Nvidia Jetson TX2.

## Getting Started
- install required packages (mostly over pip)
- clone this repository
- download and setup the COCO Dataset: `setup_coco.py`
- inside `coco.py` subclass `Config` (defined in `config.py`) and change model params to your needs
- train `mobile mask r-cnn` on COCO with: `train_coco.py`
- evaluate your trained model with: `eval_coco.py`
- do both interactively with the notebook `train_coco.ipynb`
- if you face killed kernels due to memory errors, use `bash train.sh` for infinite training
- visualize / control training with tensorboard: `cd` into your current log dir and run: <br />
`tensorboard --logdir="$(pwd)"`
- inspect your model with `notebooks/`: <br />
`inspect_data.ipynb`,`inspect_model.ipynb`, `inspect_weights.ipynb`
- convert keras h5 to tensorflow .pb model file run: <br />
`python helper/keras_to_tensorflow.py -input_model_file saved_model_mrcnn_eval -output_model_file model.pb -num_outputs=7`
<br />
<br />

## Requirements
- numpy
- scipy
- Pillow
- cython
- matplotlib
- scikit-image
- tensorflow>=1.3.0
- keras>=2.1.5
- opencv-python
- h5py
- imgaug
- IPython[all]
- pycocotools

[Original Matterport README](https://github.com/matterport/Mask_RCNN/blob/master/README.md)
