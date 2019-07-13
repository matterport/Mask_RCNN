# Skin Segmentation using Mask-RCNN

This is an extension of the applications of semantic segmentation to skin detection. Here I train the model on my own skin dataset along with other publicly available datasets and train the network. 
The networks gives 94.04 % skin pixel accuracy when tested on the 80 images of [Pratheepan](http://cs-chan.com/downloads_skin_dataset.html), 100 Images from FSD dataset sent by the authors and 100 images from the customized dataset.
All the dataset will be soon made publicly available.


## Demo
<p align="center">
<img src="https://github.com/anirudhtopiwala/Umbilicus_Skin_Detection/blob/master/Mask_RCNN/assets/skinseg-MaskRcnn.gif">
</p>


## Run Jupyter notebooks
Open the `skin.ipynb` notebook to explore the code and visualize the results step by step.

## Downloading the Trained model
To test your own images, you can download the trained model using the following instructions:
```
pip install gdown
gdown  https://drive.google.com/uc?id=1j49R6MPfba-LRV40kYgBcuEsLNUipm0S
```

## Live Skin Segmentation
To run the model on images acquired by your web cam, download and set the model path and then run:
```
python3 skin_live.py
```
