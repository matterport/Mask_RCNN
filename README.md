# Understanding cartoon emotion using integrated deep neural network on large dataset

This is an implementation of Mask R-CNN on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of a cartoon in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

![image](https://user-images.githubusercontent.com/44928935/131334979-fc5af02e-1cbc-47c9-a29d-02c30b74ae7c.png)

The current work deals with recognizing emotions from facial expressions of cartoon characters. The proposed integrated DNN approach, trained on a large dataset consisting of animations for both the characters (Tom and Jerry), correctly identifies the character, segments their face masks, and recognizes the consequent emotions with an accuracy score of 0.96. The approach utilizes Mask R-CNN for character detection, segmentaion and state-of-the-art deep learning models, namely ResNet-50, MobileNetV2, InceptionV3, and VGG 16 for emotion classification.

* The repository includes:

    * Source code of Cartoon face segmentaion using Mask-RCNN.
    * Dataset used to train Mask-RCNN model. [Cartoon Dataset](https://github.com/Shubhi3199/Mask_RCNN/releases/tag/v1)
    * Pre-trained weights of the trained Cartoon Mask-RCNN model. [Releases](https://github.com/Shubhi3199/Mask_RCNN/releases/tag/v1)
    * Jupyter notebooks to visualize the detection pipeline at every step. [Refer Code here](https://github.com/Shubhi3199/Mask_RCNN/tree/cartoon_segmentaion/samples/characters)
 
 
<h2 align = "center"> Proposed Workflow </h2>
<p align = "center">
    <img src = "https://user-images.githubusercontent.com/44928935/131338449-3b699970-d839-4903-aa07-b2e4d289f720.png" />
</p> 


<h2 align = "center"> Getting started </h2>

### *Step1 : Training Mask-RCNN model on custom dataset*
A training dataset has been generated for the Mask R-CNN model, which is later used to classify and segment Tom & Jerry’s faces from the input frames. For preprocessing, each data frame is augmented with a JSON file that stores the frame name, cartoon character name, and the X–Y coordinates of the corresponding cartoon face. The X–Y coordinates of the face were marked using a labeling tool, i.e., VGG Image Annotator (VIA).5 The marked regions were Tom’s face and Jerry’s face. The Mask R-CNN model learns this Region of Interests through the X–Y coordinates of cartoon character’s faces marked through VIA tools.
<p align = "center"> 
    <img src = "https://user-images.githubusercontent.com/44928935/131341067-7e68f2bb-bd96-4333-bbeb-bcf0973a2230.png" />
</p> 


### *Step2 : Character face detection using Mask R-CNN*
In this project Mask R-CNN is used for character face detection, which separates the foreground–background pixels from each other using a bounding box that segments the face. The model takes images or videos as input to extract masks of Tom’s face or Jerry’s face out of it.
<p align = "center"> 
    <img src = "https://user-images.githubusercontent.com/44928935/131338324-ff485ab4-1df7-482b-ac8b-0f366eee705f.png" />
</p>  

### *Step3 : Emotion classification using transfer learning and fine-tuning*
Further, the masks produced by Mask-RCNN are given as input to Convolution neural network model VGG16. Then, the emotion of a particular character is recognized with an emotion confidence score (which is recognized emotion probability of the respective character) scaled in the range of 0 to 1.

<p align = "center"> 
    <img src = "https://user-images.githubusercontent.com/44928935/131339875-11cc7480-5f9b-46fc-85f1-9951e46d3e75.png" />
</p>  

### *Results obtained*
<p align = "center"> 
    <img src = "https://user-images.githubusercontent.com/44928935/131340238-b0ca955c-2b76-40df-8c86-0df0a418dfeb.png" />
</p> 

#### For a detailed explanation of the project [Refer Our Research Paper](https://link.springer.com/article/10.1007/s00521-021-06003-9)

## Citation
Jain, N., Gupta, V., Shubham, S. et al. Understanding cartoon emotion using integrated deep neural network on large dataset. Neural Comput & Applic (2021). https://doi.org/10.1007/s00521-021-06003-9
