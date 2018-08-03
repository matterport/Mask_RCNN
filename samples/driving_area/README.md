# Driveable Area Detection Example

This is an example showing the use of Mask RCNN in a real application. In this application, we show how a Mask R-CNN model can be used to train and detect driveable regions from dashcam/car camera footage. 


![Driveable Area Detection](/assets/driving_area.gif)


## Dataset

This project uses the [BDD Dataset](http://bdd-data.berkeley.edu/). After downloading the dataset, follow the below steps.
* Ensure you have a `${DATA_DIR}/images/100k/` directory and it has `train/` and `val/` subdirectories.
* _Minor Change_ :Ensure you place all the `${DATA_DIR}/driveable_maps/color/*` files in the same folder (ie, NO training/validation/test split). Place all the masks in a single folder `${DATA_DIR}/driveable_maps/color/train/`.

Here `${DATA_DIR}` is the root/parent directory for the dataset which will be passed into the training script as an argument.


## Apply Driveable Area Filter
Apply on an image:

```bash
python3 driving_area.py detect --weights=/path/to/mask_rcnn/mask_rcnn_driveable.h5 --image=<file name or URL>
```

Apply on a video. Requires OpenCV 3.2+:

```bash
python3 driving_area.py detect --weights=/path/to/mask_rcnn/mask_rcnn_driveable.h5 --video=<file name or URL>
```


## Run Jupyter notebooks
Open the `inspect_data.ipynb` or `inspect_model.ipynb` Jupter notebooks. You can use these notebooks to explore the dataset and run through the detection pipelie step by step.

## Train the Balloon model

Train a new model starting from pre-trained COCO weights
```
python3 driving_area.py train --dataset=/path/bdd/dataset --weights=coco
```

Resume training a model that you had trained earlier
```
python3 driving_area.py train --dataset=/path/to/bdd/dataset --weights=last
```

Train a new model starting from ImageNet weights
```
python3 driving_area.py train --dataset=/path/to/bdd/dataset --weights=imagenet
```

The code in `driving_area.py` is set to train for 875k steps (500 epochs of 1750 steps each), and using a batch size of 2. 
Update the schedule to fit your needs.
