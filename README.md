# Mobile Mask R-CNN
This is a Mask R-CNN implementation with MobileNet V1/V2 as Backbone architecture to be finally able to deploy it on mobile devices such as the Nvidia Jetson TX2.
The major changes to the original [matterport project](https://github.com/matterport/Mask_RCNN) are: <br />
- [X] Add Mobilenet V1 and V2 as backbone options (besides ResNet 50 and 101) + dependencies in the model
- [X] Make the whole project py2 / py3 compatible (original only works on py3)
- [X] Investigate Training Setup for Mobilenet V1 and implement it in `coco_train.py`
- [X] Add a Speedhack to mold /unmold image functions
- [X] Make the project lean and focused on COCO + direct training on passed class names (IDs before)
- [ ] Inclue more speed up options to the Model (Light-Head RCNN)
- [X] Release a trained Mobile_Mask_RCNN Model
<br />

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
`inspect_data.ipynb`,`inspect_model.ipynb`, `inspect_weights.ipynb`,`detection_demo.ipynb`
- convert keras h5 to tensorflow .pb model file, in `notebooks/` run: <br />
`export_model.ipynb`
<br />


## Performance
Mobile Mask R-CNN trained on 512x512 input size
- 100 Proposals: 0.22 mAP (VOC) @ 250ms
- 1000 Proposals: 0.25 mAP (VOC) @ 330ms
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
