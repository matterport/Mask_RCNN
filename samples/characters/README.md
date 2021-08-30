# Tom & Jerry Cartoon character's face segmentation Example

This is an example showing the use of Mask RCNN in a real application.
We train the model to detect cartoon characters (Tom and Jerry), and then we use the generated 
face masks of the cartoon characters to classify the emotion of the cartoon character in that image into 4 classes - Happy, Sad, Angry, Surprise using VGG16 Convolution Neural Network.

## Installation
From the [Releases page](https://github.com/Shubhi3199/Mask_RCNN/releases/tag/v1) page:
1. Download `mask_rcnn_characters_0049.h5`. Save it in the root directory of the repo (the `mask_rcnn` directory).
2. Download `cartoon_dataset.zip`. Expand it such that it's in the path `mask_rcnn/datasets/characters/`.

## Run Jupyter notebooks
Open the `Inspect_TomJerry_Model.ipynb` Jupter notebook to explore the Tom-Jerry Dataset and run through the detection pipelie step by step.
Also, Open the `Tom_&_Jerry_TF_2_1_Combined.ipynb` Jupter notebook for understanding the complete flow the project consisting of Segmentation of cartoon characters face mask followed by emotion classification CNN model training. 

## Train the Tom_Jerry model

Train a new model starting from pre-trained COCO weights
```
python3 TomJerry.py train --dataset=/path/to/characters/dataset --weights=coco
```

Resume training a model that you had trained earlier
```
python3 TomJerry.py train --dataset=/path/to/characters/dataset --weights=last
```

Train a new model starting from ImageNet weights
```
python3 TomJerry.py train --dataset=/path/to/characters/dataset --weights=imagenet
```