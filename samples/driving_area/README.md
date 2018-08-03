# Driveable Area Detection Example

This is an example showing the use of Mask RCNN in a real application. In this application, we show how a Mask R-CNN model can be used to train and detect driveable regions from dashcam/car camera footage. 


![Driveable Area Detection GIF](/assets/driving_area.gif)

A high definition video of the above gif can be found [here](https://www.youtube.com/watch?v=NFeXQhzYN8Q&feature=youtu.be).

A few of the sample inferences are shown below.

![Driveable Area Detection 1](/assets/driving_area_1.png)

![Driveable Area Detection 2](/assets/driving_area_2.png)

![Driveable Area Detection 3](/assets/driving_area_3.png)

![Driveable Area Detection 4](/assets/driving_area_4.png)


## Dataset

This project uses the [BDD Dataset](http://bdd-data.berkeley.edu/). After downloading the dataset, follow the below steps.
* Ensure you have a `${DATA_DIR}/images/100k/` directory and it has `train/` and `val/` subdirectories.
* _Minor Change_ :Ensure you place all the `${DATA_DIR}/driveable_maps/color/*` files in the same folder (ie, NO training/validation/test split). Place all the masks in a single folder `${DATA_DIR}/driveable_maps/color/train/`.

Here `${DATA_DIR}` is the root/parent directory for the dataset which will be passed into the training script as an argument.

* _Note_:The BDD Dataset is explicitly non-commercial/research only license and all the above images/assets are therefore licensed under the same terms. Please refer to the original license when downloading the dataset. 

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
Open the [inspect_driving_model](inspect_driving_model.ipynb) Jupter notebook. You can use the notebooks to run through the detection pipelie step by step.

## Train the model

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
