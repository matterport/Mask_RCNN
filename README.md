Technion - Israel institute of technology

CS 234329 - Project in image processing and analysis

18 April 2019

###This Document is still in process
Abstract
========

This Project tackles the task of segmentation and indexation on cucumber
plant parts. The approach is to implement an end to end workflow to
partially annotated and relatively small datasets.

Mask RCNN paper from facebook research has proven very efficient in this
task even on small datasets. Therefore the focus is maximizing the
effectiveness of a problematic dataset on different levels. This reports
discusses partially annotated datasets, and inconsistent annotation
styles.

The conclusion tends towards a trade-off between investing in the
dataset quality and using augmentation acrobatics to quickly use the
existing data. There is no doubt that consistent fully annotated real
pictures fully will get the most out of the network.

Results (2 or 3 best segmentation images) and conclusions

Introduction
============

Mask-CNN is widely used and it’s high performance is proved on
large-scale datasets (COCO). But what about small datasets? Augmented?
Synthetic?

\[show here achievements in a brief way, basically all the titles and
subtitles

Authors: Simon Lousky, Asher Yartsev, Or Shemesh

Supervisors: Alon Zvirin, Yaron Honen, Dima Kuznichov

**Table of Contents**
=====================

System Description
==================

Mask R-CNN in a nutshell
------------------------

It is most easy to think of this network as a tree where the nodes are
DNN components, each with a simple and specific task, and describe the
relationship between those components. The first block is a CNN (call it
backbone network) which should be some state of the art feature
extractor like ResNet.

The second component would be the RPN (Region Proposal Network) Which
shares convolutional layers with the backbone using a FPN (Feature
Pyramid Network). FPN improves region proposal and is more accurate than
using only the last feature map layer.

The third node is a FCN (Fully Connected) that performs both regression
on the bounding boxes and classification inside the regions.

The last one is the masking CNN part which is basically a binary
classifier that indicates for each pixel in the region whether it is
part of an object or not.

More Mask R-CNN implementation details are explained in a later chapter.

  ![](./media/image69.png){width="6.619792213473316in" height="2.2335476815398074in"}
  -------------------------------------------------------------------------------------
  [Figure 1](#figur_mrcnn_graph) - Schematic of Mask R-CNN architecture

Our dataset
-----------

The dataset was 4 sets of 6000 x 4000 pictures, accompanied by a .json
file in the COCO format[^1] .Our goal was to successfully detect four
different classes, preferably doing it in a single forward pass on a
picture. All four classes were four different parts of a cucumber plant:

1.  Fruits (aka cucumber) - 101 Pictures

2.  Leaves - 140 pictures

3.  Flowers - 131 pictures

4.  Stems - 481 pictures

Each class having its own dataset, they differed both in quality and
quantity.

Training this Network assumes full supervision of all objects, Therefore
not having one combined dataset is the main issue, given the need to
inference through a single network for all our classes. On the other
hand hundreds of picture (one would argue that tens of pictures) is a
reasonable amount for a pre-trained Mask RCNN model, both in terms of
numbers and diversity.

Dataset Inconsistencies
-----------------------

Three major types of inconsistencies made our dataset challenging.

### Occlusion management

The first is occlusion management. In some cases the masks contain
occluded parts, and in others the occlusion is omitted. For example in
[figure 2](#fig_occlusion), the stem that hides the blue left cucumber
is annotated as part of it, but the cucumber in the middle of the
picture is separated by a stem. Moreover it is worth noticing that COCO
dataset is consistent regarding the occlusion matter. See [figure
2](#fig_occlusion) - The man’s hands are hiding the surfboard but are
not marked as part of it.

### Multiple polygons for single object

The second anomaly is annotations of different parts of the same object
as different objects. See the middle pink cucumber in [figure
2](#fig_occlusion), the yellow color indicates that it is annotated as a
new fruit. In this case as well the COCO annotation style permits
multiple polygons for a single object. See [figure 2](#fig_occlusion) -
The car in the background is made of two separate polygons, or the three
parts of the surfboard that are annotated as the same object.

  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  ![](./media/image74.png){width="1.896599956255468in" height="1.896599956255468in"}![](./media/image73.png){width="1.896599956255468in" height="1.896599956255468in"}![](./media/image75.png){width="1.571340769903762in" height="2.766925853018373in"}
  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  [Figure 2](#figur_occlusion) - Annotation of occluded objects.
  
  Left image shows the typical anomalies found in the original dataset.
  
  Middle Image is the desired annotation for such image.
  
  The Right image is an example from MS-COCO (image id 246120).
  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

###  Under supervised datasets

The third inconsistency type is low percentage of annotated objects per
image in our dataset. Like previously said, the network assumes full
supervision over the objects, therefore having relevant but unannotated
objects, is a serious pitfall in using the picture as ground truth.
Let’s assume we wish to detect all mature foreground leaves. In [figure
3](#fig_full_supervision), you can see only one leaf is annotated out of
many leaves that are clearly visible, compared to a non perfect but yet
significantly better annotated image on the right ([figure
4](#fig_apples)), where 13 apples out of 16 are annotated.

  ![](./media/image58.png){width="2.314912510936133in" height="1.5166666666666666in"}                                           ![](./media/image77.png){width="1.9104166666666667in" height="1.9104166666666667in"}
  ----------------------------------------------------------------------------------------------------------------------------- --------------------------------------------------------------------------------------
  [Figure 3](#figur_full_supervision) - Annotation of frequent objects images. Left annotated image from our dataset compared   [Figure 4](#figur_apples) - Image from MS-COCO (image id 239565)

Towards data enhancement
========================

First steps
-----------

The direction we were thrown at regarding the dataset issue was
synthetic data creation. Based on the work previously made by the lab,
we were about to train the model with pictures of randomly dispersed
objects on neutral backgrounds, generated at training time ([figure
5](#fig_synthetic_avocado) and [figure 6](#fig_avocado_inference)). This
technique seemed promising and we were intrigued by the question whether
this approach would yield results as good as real, fully annotated
datasets. This question lead to some work we discuss later on.

  ![](./media/image76.png){width="2.3958333333333335in" height="2.3854166666666665in"}   ![](./media/image62.png){width="2.4270833333333335in" height="2.3854166666666665in"}
  -------------------------------------------------------------------------------------- --------------------------------------------------------------------------------------
  [Figure 5](#figur_synthetic_avocado) - Synthetic image                                 [Figure 6](#figur_avocado_inference) - Inference after synthetic training

Repairing the dataset
---------------------

Looking at the final goal that was at stake, we were determined in any
way possible to make the best out of what we got. We knew were going to
want to play around with the data, change resolutions, augment the
pictures, maybe even fix some of the pitfalls in the dataset. For this
reason we searched for an open source tool that could help us visualize
and manipulate the dataset. The best option we found was called
*COCO-annotator*[^2], it was intuitive, easy enough to configure and
bring up locally. This tool gave us the ability to manipulate, add
annotations and preview the dataset immediately, but we needed some
additional features we later developed:

1.  Merger Tool - This tool was intended to connect few polygons that
    > the original dataset was considering multiple objects. See [figure
    > 2](#fig_occlusion)

2.  Stylus and Ipad related features - In order to get the most out of
    > the tool, we thought adapting it to a stylus enabled touch device
    > would make the experience of annotating and correcting a bad
    > dataset much easier. The obvious additions were:

    a.  Zoom in out

    b.  Move around

    c.  Mouse events changed touch compatible events

3.  Performance improvement - The original repository had a background
    > task draining all the resources of the server and never saving the
    > work done, we fixed this issue as well.

  ![](./media/image78.png){width="6.114583333333333in" height="3.6666666666666665in"}
  -------------------------------------------------------------------------------------
  [Figure 7](#figur_coco_annotator)- Our altered version of COCO Annotator

Having the final goal in mind, we used the newly improved coco-annotator
and did some work on the original dataset.

One additional thing to note is that in any step we took regarding
manipulation of the dataset, including data augmentation we later talk
about, we insisted on having a way back to the original format of the
data set. That is - A simple directory with a bunch of picture and a
single .json file with all the annotations.

Synthetic datasets
------------------

The first step in image generation was singling out objects out of the
pictures, using the segmentation coordinates found in the .json file.
This was done using a semi-smart script that could find any object, cut
around its edges, normalize its size, and apply transparency to its
background. In order to get minimal manual work on the resulting object
bank, the script

Iterated over all the segmentations in the annotations file, filtered
out objects with irrelevant size or bad width/height ratio, resized all
of them and saved them in a new directory, one for each class.

### Object filtering

As stated before, objects were filtered using simple logic. One
parameter was the size of the object compared to the picture, we fixed a
minimum size of 10% of the pictures’s height. The Second parameter was
the minimal height:width aspect ratio of an object. Fruits ratio was
fixed at 3:1, leaves at 1:1, stems at 4:1. Another matter to address was
occluded objects which could yield partial objects. For this we only
singled out objects made of one polygon. Using this simple filtering
algorithm reduced the manual work by an order of magnitude. Although you
could arguably use the direct output of this script to train the network
and get good results, we decided to manually filter the data even more
carefully.

  ![](./media/image17.png){width="0.3906255468066492in" height="1.017442038495188in"}![](./media/image29.png){width="0.5052088801399826in" height="0.9983880139982502in"}![](./media/image48.png){width="0.3658737970253718in" height="0.9843755468066492in"}![](./media/image23.png){width="0.3130577427821522in" height="0.9739588801399826in"}![](./media/image30.png){width="0.7552088801399826in" height="0.9553838582677165in"}![](./media/image12.png){width="0.7559448818897638in" height="0.9427088801399826in"}![](./media/image36.png){width="1.109375546806649in" height="0.9685017497812773in"}![](./media/image16.png){width="0.9947922134733158in" height="0.9601902887139108in"}
  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  [Figure 8](#figur_objects) - Sample of cut and filtered objects

### Enhanced blending

One might argue that incorporated objects in a synthetic pictures can
become easy targets for a deep neural network. The sharp edges of the
cut object definitely break the pattern of any background and therefore
the network might focus on this feature. In an attempt to imitate real
life images, we blurred the mask contour to make the edges of the RGB
picture blend with the background more realistically (See [figure
9](#fig_blurred_mask)). This theory is later put to the test.

  ![](./media/image49.png){width="1.9612981189851268in" height="2.2295603674540683in"}                                                               ![](./media/image56.png){width="3.650168416447944in" height="1.7864588801399826in"}
  -------------------------------------------------------------------------------------------------------------------------------------------------- --------------------------------------------------------------------------------------------------------------------------
  [Figure 9](#figur_blurred_mask)- Blurred mask makes the RGB content fade to the background RGB content, without blurring the RGB content itself.   [Figure 10](#figur_enhanced_blending)- on the left a sharp edge, on the right a leaf edge with enhanced blending applied

### Object scenery

One other aspect of generating an image is the composition of the
pictures. How many leaves are there, how are they dispersed, how big are
they compared to the flowers and to the cucumbers etc… These hyper
parameters are all concentrated in a configuration file and are as
follow:

1.

2.

  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  ![](./media/image27.png){width="1.40625in" height="1.4027777777777777in"}
  
  Final image
  ----------------------------------------------------------------------------------- --------------------------------------------------------------------------- --------------------------------------------------------------------------- ---------------------------------------------------------------------------
  [*Figure*](#figur_generated_sample) - Sample image from a mixed synthetic dataset
  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Real datasets
-------------

In spite that the original work was about training the network using the
generated pictures technique, we found ourselves interested in comparing
the results with a fully supervised training set of original pictures.
This dataset would also need to be augmented in the DNN fashion way.
Here we describe the treatments made offline (not in during training
time) to improve the dataset.

We decided to enlarge our dataset by using a stochastic augmentation
method, that can sample altered pictures from any dataset, applying
directional flips, skews, brightness and contrast shifts by some degree
of randomness. To do so, we used the Python module *Augmentor*[^3]. In
this process we tried to preserve the quality of the dataset by erasing
nonsense annotations that rather looked like artifacts on the altered
pictures. Objects smaller than 150 pixels in a 1536 x 1024 image and
annotations of objects with extreme aspect ratios were removed. Pictures
left with no annotations were removed as well.

  ![](./media/image79.png){width="6.270833333333333in" height="4.285416666666666in"}
  ---------------------------------------------------------------------------------------------------------------------------------------
  [Figure 11](#figur_augmentor) - Example of new augmented images generated from the same original image (IMG\_2718.jpg) in the center.

Mask R-CNN Train and Implementation aspects
===========================================

Mask R-CNN on Tensorboard
-------------------------

We invested some effort to redesign the original *matterport/Mask\_RCNN*
implementation to take advantage of Tensorboard’s visual power, since
original code wasn’t written with this design in head.

That way we, and others following us, will be able to analyze the net
structure with confidence:

![](./media/image61.png){width="6.878472222222222in"
height="2.9895833333333335in"}

\[note\] full png can be found in project repository on GitHub.

To emphasize the need, this is a very small part of the default
generated graph which make the net impossible to track and debug when
needed:

![](./media/image70.png){width="2.463542213473316in"
height="6.274662073490814in"}

Mask R-CNN flowchart
--------------------

During the first steps and last steps of our project we realized that
fully understanding the architecture of Mask R-CNN is helping us focus
on hyperparameters better. Unfortunately no satisfactory documentation
was found, and TensorFlow showed way too much graphic “noise” that
clouded the big picture. So with patience and full correspondence to the
implementation we leaned on, we formulated a flowChart
([*figure*](http://fig_flowchart)) to help us and others in the future.

  ![](./media/image68.jpg){width="6.114583333333333in" height="3.5989588801399823in"}
  -------------------------------------------------------------------------------------
  ![](./media/image22.jpg){width="6.114583333333333in" height="3.5781255468066493in"}

### [Figure 12](#figur_flowchart)

Monitoring through callbacks
----------------------------

The implementation we worked with, used embedded TensorFlow callbacks,
this way we could graphically monitor each Loss function. ([Figure
13](#fig_losses))

  ![](./media/image64.png){width="2.9791666666666665in" height="2.236111111111111in"}   ![](./media/image72.png){width="2.9791666666666665in" height="2.2083333333333335in"}
  ------------------------------------------------------------------------------------- --------------------------------------------------------------------------------------
  ![](./media/image66.png){width="2.9791666666666665in" height="2.236111111111111in"}   ![](./media/image71.png){width="2.9791666666666665in" height="2.236111111111111in"}

[Figure 13](#figur_losses)

\[note\] the graphs were taken from small experimental training
procedures, in which we just tested an innovative training technique on
generated data. Every few epochs we generated totally new data,
gradually incrementing the number and scale of generated objects. We
were guided by the intuition that a good practice to avoid overfit is
generating more complicated dataset every few epochs. In other words,
each time the model is performing well on the dataset we challenge it
harder.

In our research process, during first training phazes, we noticed that
many times our training session reached a plateau in terms of minimizing
the Loss.

We put efforts to support more TensorFlow Callbacks allowing
visualization of weight distribution in each layer during training.

Unfortunately, results can only be shown with the initialization
weights, and their evolution over time could not be monitored.

We eventually found out that due to poor implementation design pursuing
this thread would be too time consuming.

Hyper-Parameters Analysis
-------------------------

We believe it’s vital to elaborate on optional configurations that
affect this network. This section expresses the vast domain of possible
twicks and fine-tunings.

### RPN architecture

-   RPN\_ANCHOR\_SCALES: We put some thoughts in the size we want our
    > anchors to be, since a better set of anchor sizes, one that
    > statistically fits most of the objects, should lead to a slightly
    > better convergence of *rpn\_bbox\_loss*.

-   RPN\_ANCHOR\_RATIOS: Ratios are sometimes also a consideration to
    > improve bbox\_loss, for example, training on stems should require
    > high and thin anchors. For some implementational reasons, adding
    > ratios instead of replacing them leads to dimension mismatches.
    > Pursuing this thread could be interesting since thin objects and
    > thin parts in general seem challenging according to our experience
    > and to some related topics on the internet.

-   RPN\_NMS\_THRESHOLD: Intersection percent over which two ROIs are
    > considered one single object and only one is kept. We can’t
    > determine exactly what is better: Values closer to 1 filter out
    > best matches, therefore converge faster at the beginning, or
    > smaller values that slow down convergence at the beginning but
    > might produce better results since the learning curve is slower.

-   RPN\_TRAIN\_ANCHORS\_PER\_IMAGE: The final number of ROIs to
    > consider on a single forward pass. The ROIs are randomly selected
    > from the result of the RPN. It’s important to realize that this
    > HyperParameter is coupled with RPN\_NMS\_THRESHOLD since, the
    > higher the threshold, the more diverse ROI proposal is. High NMS
    > will be more likely to choose dispersed anchors to train on. This
    > should represent better the underlying distribution of objects in
    > the scene.

### Masking network

-   USE\_MINI\_MASK: This boolean is for faster performance at the cost
    > of masks accuracy.\
    > Empirically we can state that turning on this variable had a
    > significantly bad effect on our masks quality.We always prefered
    > to turn it off.

### General

-   LEARNING\_RATE: We explored few approaches:

    -   Extra small learning rate (micro scale)

    -   A big learning rate(\~0.1)

    -   Incremental approach where in each epoch we reduce learning
        > rate.

> We’ve found out that high learning rates lead to poor train results.
> Extra small learning rate had no significant benefit over a decaying
> learning rate that starts from 0.001. Therefore, we decided to move on
> with our incremental approach.

Workflow
========

Here we describe the flow of actions from getting a problematic dataset
to training and testing the model.

Data Viability
--------------

1.  Quickly inspect the json to check it is formatted correctly. In our
    > case some of the files had the wrong order of segmentation values.
    > We used a python script[^4] to correct the data.

2.  Visually check the sanity of the data by opening it in our
    > *COCO-Annotator*[^5]*.*

3.  **Correct the data if needed**. If your dataset is small, and you
    > find a significant pitfall, the right tools will get you through
    > it very fast, and save you time later trying to bypass those.

Data Organization
-----------------

1.  Split the dataset to train and validation sets, we made a script for
    > this as well.

2.  Resize the dataset pictures if needed. We could not train on our
    > 4000x6000 original pictures therefore we resized all of them to
    > 1536x1024 and 768x512 using the *altered Augmentor*[^6].

3.  Extract the single objects for synthetic image generation. Keep
    > track of object created from the training set and those created
    > from the validation set. This step as well is assisted by a
    > script. This step is followed by an optional manual filtering
    > step.

4.  Keep the dataset organized in a practical fashion. We used the
    > layout you can see in [figure 14](#fig_directory_layout) to
    > organize the datasets. Take in consideration that the data
    > generating class we created assumes this layout in order to
    > properly work.

Data Augmentation
-----------------

1.  For real supervised dataset, use the altered augmentor to sample a
    > dataset big enough. We sampled 2000 pictures from every original
    > dataset.

2.  For generated dataset, tune the object scenery parameters according
    > to your understanding. The parameters are depicted in the chapter
    > about synthetic datasets.

  ![](./media/image65.png){width="2.948017279090114in" height="2.127603893263342in"}
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  [Figure 14](#figur_directory_layout) - Generated layout has no resolution layer since all objects and backgrounds are compatible with every synthetic image resolution.

Training
--------

1.  Read the repository readme and just start your training!

2.  Have fun!

Experiments and Results
=======================

Metrics
-------

### IoU

Intersection over union, a method to quantify the percent overlap
between target mask and prediction output.

### Plot overlap matrix

Show number of predictions vs actual instances

Show prediction confidence

Show IoU for each prediction

### Precision recall graph

sorted Precision\* as function of recall\
**Precision** - Number of true positives over total
classifications$\ \frac{\text{True\ Positive}}{True\ Positive\  + \ False\ Positive}$\
**Recall** - Number of true positives over total true instances
$\frac{\text{True\ Positive}}{True\ Positive\  + \ False\ Negative}$

Fruit only - real vs synthetic
------------------------------

Definition of class - foreground mature fruits, occlusion omitted.

  **Mean AP - IoU=50**
  ---------------------- ---------------- ------------------------- --------------- ---------- ---------- ----------
  **Type**               **Resolution**   **Train set**             **Original **   **4000**   **Mean**
  **Synthetic **         **512**          **Fruit only**            0.38            0.07       0.36
                         **1024**         **Fruit only**            0.42            0.09       
                         **1024**         **Mixed**                 0.29            0.03       
  **Real**               **512**          **Fruit only (2000)**     0.53            0.51       **0.65**
                         **1024**         **Fruit only (2000)**     0.70            0.56       
                         **1024**         **Fruit only (10,000)**   0.71            0.70       

  **Mean AP - IoU=20**
  ---------------------- ---------------- ------------------------- --------------- ---------- ----------- ----------
  **Type**               **Resolution**   **Train set**             **Original **   **4000**   **Mean**
  **Synthetic **         **1024**         **Fruit only**            0.44            0.09       0.37
                         **1024**         **Mixed**                 0.30            0.03       
  **Real**               **1024**         **Fruit only (2000)**     0.73            0.56       **0.725**
                         **1024**         **Fruit only (10,000)**   0.72            0.70       

Side note about synthetic data and high resolution testing: Earlier
models trained with much bigger object sizes reacted very much better on
the high resolution pictures. Nevertheless the real dataset training set
makes the model much more robust to resolution change.

  Real 1024                                                                 Synthetic 1024                                                                         Ground truth
  ------------------------------------------------------------------------- -------------------------------------------------------------------------------------- -------------------------------------------------------------------------
  ![](./media/image18.jpg){width="2.0833333333333335in" height="1.375in"}   ![](./media/image40.jpg){width="2.0833333333333335in" height="1.375in"}                ![](./media/image59.jpg){width="2.0833333333333335in" height="1.375in"}
  ![](./media/image15.jpg){width="2.0833333333333335in" height="1.375in"}   ![](./media/image8.jpg){width="2.0833333333333335in" height="1.375in"}                 ![](./media/image38.jpg){width="2.0833333333333335in" height="1.375in"}
  ![](./media/image37.jpg){width="2.0833333333333335in" height="1.375in"}   ![](./media/image35.jpg){width="2.0833333333333335in" height="1.375in"}                ![](./media/image10.jpg){width="2.0833333333333335in" height="1.375in"}
  ![](./media/image33.jpg){width="2.0833333333333335in" height="1.375in"}   ![](./media/image41.jpg){width="2.0833333333333335in" height="1.3888888888888888in"}   ![](./media/image55.jpg){width="2.0833333333333335in" height="1.375in"}

Leaves only - real vs synthetic
-------------------------------

Definition of class - foreground mature leaves, occlusion omitted.

The original dataset is only partially supervised making training on a
real dataset problematic. We could just ignore the fact that it is
partially supervised but the problem would then be testing the results.
Testing is tricky because the test set have to be be fully supervised in
order to give a correct precision and recall grades.

For this purpose we used the COCO-annotator to supervise a very small
dataset, including a tiny test set of two pictures.

(mini dataset fully supervised) vs synthetic (full dataset) vs real
(partially supervised)

  **Mean AP - IoU=50**
  ---------------------- ---------------- ---------------------- --------------- ---------- ---------- ----------
  **Type**               **Resolution**   **Train set**          **Original **   **4000**   **Mean**
  **Synthetic **         **512**          **Leaf only**          1.0             0.0        0.33
                         **1024**         **Leaf only**          0.0             0.0        
                         **1024**         **Mixed**              0.0             0.0        
  **Real**               **512**          **Leaf only (2000)**   1.0             0.0        **1.0**
                         **1024**         **Leaf only (2000)**   1.0             0.77       

  **Mean AP - IoU=20**
  ---------------------- ---------------- ---------------------- --------------- ---------- ---------- ----------
  **Type**               **Resolution**   **Train set**          **Original **   **4000**   **Mean**
  **Synthetic **         **1024**         **Leaf only**          0.48            0.0        0.335
                         **1024**         **Mixed**              0.19            0.0        
  **Real**               **1024**         **Leaf only (2000)**   1.0             0.77       **1.0**

  Synthetic 1024                                                            Real 1024                                                                 Ground truth
  ------------------------------------------------------------------------- ------------------------------------------------------------------------- -------------------------------------------------------------------------
  ![](./media/image67.jpg){width="2.0833333333333335in" height="1.375in"}   ![](./media/image11.jpg){width="2.0833333333333335in" height="1.375in"}   ![](./media/image52.jpg){width="2.0833333333333335in" height="1.375in"}
  ![](./media/image19.jpg){width="2.0833333333333335in" height="1.375in"}   ![](./media/image57.jpg){width="2.0833333333333335in" height="1.375in"}   ![](./media/image44.jpg){width="2.0833333333333335in" height="1.375in"}

\
-

Stems only - real vs synthetic
------------------------------

Definition of class - Foreground main stem, occluded location included.

  **Mean AP - IoU=50**
  ---------------------- ---------------- ---------------------- --------------- ---------- ---------- -----------
  **Type**               **Resolution**   **Train set**          **Original **   **4000**   **Mean**
  **Synthetic **         **512**          **Stem only**          0.0             0.01       0.015
                         **1024**         **Stem only**          0.03            0.01       
  **Real**               **512**          **Stem only (2000)**   0.05            0.01       **0.08**
                         **1024**         **Stem only (2000)**   0.11            0.08       

  **Mean AP - IoU=20**
  ---------------------- ---------------- ---------------------- --------------- ----------
  **Type**               **Resolution**   **Train set**          **Original **   **4000**
  **Synthetic **         **1024**         **Stem only**          0.24            0.22
  **Real**               **1024**         **Stem only (2000)**   0.62            0.54

  Real 1024                                                                              Ground truth
  -------------------------------------------------------------------------------------- --------------------------------------------------------------------------------------
  ![](./media/image5.jpg){width="3.1527777777777777in" height="2.0833333333333335in"}    ![](./media/image7.jpg){width="3.1666666666666665in" height="2.0833333333333335in"}
  ![](./media/image25.jpg){width="3.1527777777777777in" height="2.0833333333333335in"}   ![](./media/image4.jpg){width="3.1666666666666665in" height="2.0833333333333335in"}
  ![](./media/image45.jpg){width="3.138888888888889in" height="2.0833333333333335in"}    ![](./media/image26.jpg){width="3.1666666666666665in" height="2.0833333333333335in"}

The model trained on the real dataset is not particularly wrong with its
decisions, but the limited aspect ratio seems to be responsible for
splitting the stems. Split detections are not counted as true positives
thus the very small precision.

On the other hand the model trained on synthetic data does not perform
well at all on stems.

Flowers only - real vs synthetic
--------------------------------

Definition of class - Foreground main stem, occluded location included.

  **Mean AP - IoU=50**
  ---------------------- ---------------- ------------------------ --------------- ---------- ---------- -----------
  **Type**               **Resolution**   **Train set**            **Original **   **4000**   **Mean**
  **Synthetic **         **512**          **Flower only**          0.0             0.01       0.015
                         **1024**         **Flower only**          0.03            0.01       
  **Real**               **512**          **Flower only (2000)**   0.05            0.01       **0.08**
                         **1024**         **Flower only (2000)**   0.11            0.08       

  **Mean AP - IoU=20**
  ---------------------- ---------------- ------------------------ --------------- ---------- ---------- --
  **Type**               **Resolution**   **Train set**            **Original **   **4000**   **Mean**
  **Synthetic **         **1024**         **Flower only**          0.30            0.12       
                         **1024**         **Mixed **               0.14            0.04       
  **Real**               **1024**         **Flower only (2000)**   0.64            0.56       

  Synthetic 1024                                                            Real 1024                                                                 Ground truth
  ------------------------------------------------------------------------- ------------------------------------------------------------------------- --------------------------------------------------------------------------------------
  ![](./media/image24.jpg){width="2.0833333333333335in" height="1.375in"}   ![](./media/image51.jpg){width="2.0833333333333335in" height="1.375in"}   ![](./media/image43.jpg){width="2.0833333333333335in" height="1.375in"}
  ![](./media/image6.jpg){width="2.0833333333333335in" height="1.375in"}    ![](./media/image32.jpg){width="2.0833333333333335in" height="1.375in"}   ![](./media/image20.jpg){width="2.0833333333333335in" height="1.3611111111111112in"}
  ![](./media/image34.jpg){width="2.0833333333333335in" height="1.375in"}   ![](./media/image39.jpg){width="2.0833333333333335in" height="1.375in"}   ![](./media/image28.jpg){width="2.0833333333333335in" height="1.375in"}

Flowers

Mixed - enhanced blending vs sharp blending
-------------------------------------------

  **Mean AP - IoU=50**
  ---------------------- ------------------- --------------- --------------------- ----------------------- ------------
                         **Cucumber only**   **Stem only**   **Mixed on flower**   **Mixed on cucumber**   **Mean**
  **Erode and Blur**     0.42                0.03            0.07                  0.29                    **0.2025**
  **No blending**        0.41                0.001           0.01                  0.22                    0.16025

  **Mean AP - IoU=20**
  ---------------------- ------------------- --------------- --------------------- ----------------------- ------------
                         **Cucumber only**   **Stem only**   **Mixed on flower**   **Mixed on cucumber**   **Mean**
  **Erode and Blur**     0.45                0.24            0.14                  0.30                    **0.2825**
  **No blending**        0.43                0.12            0.02                  0.23                    0.2

Blur and erosion properties clearly influence the result, and should be
optimized.

Mixed - team synthetic vs team real
-----------------------------------

These two network are not exactly comparable since the second is really
made of 4 networks, all combined to a single output, where the synthetic
is the result of a single forward pass.

Test on train resolution

Test on 4000 resolution (target)

IoU=50

Conclusions
===========

Lack of robustness is due to

Since it is hard to imitate real images it is higly probable that our
algorithms created patterns the model.

Which directly influenced the responsiveness of RPN to specific feature
maps in the pyramid.

Due to limited resources, inference configuration for 4000 test images
crops the images to patches of 2048 x 2048.

Training on higher resolution effect

Training on large dataset vs small dataset

Training on synthetic vs real

-   Due to hardware limitations, on inference, Mask-RCNN has to apply
    > Image-Patching approaches, which causes big objects to partially
    > appear in each patch, thus, get painted partially few times - this
    > effect reduces Mean-AP drastically, since many maskings don’t even
    > cross the 50% IoU threshold.

-   A biased hypothesis occured, during our research process, which
    > states that a mixed-object model performs worse than a
    > single-object model. By analyzing our research containers and
    > observing generated dataset for the Model, it dawned on us that
    > our generator generates up to 20 objects, no matter who they are.
    > Thus **the reacher the object distribution got, the less actual
    > objects from each type appeared**.

Further Work and Suggestions
============================

-   Smart synthetic plants - spreading leaves and cucumbers around a
    > stem

-   Transfer learning of masks between classes (e.g for flowers masking)

-   Real time inference - run real time inferences while keeping track
    > of object indexation. Keep track of evolution for each leaf or
    > cucumber as well.

-   LSTM or any RNN concept to improve object detection (flower turns to
    > cucumber, stems grows up, etc)

-   Use a 3D engine platform to generate data. Create infinite angles,
    > curvature and lighting of leaves, using a single leaf texture.

-   Background generative models. Generate backgrounds instead of using
    > a finite bank of backgrounds. Possibly this would better mimic
    > real supervised pictures.

-   Investigate the loss of the model for very thin objects and thin
    > parts, and try to propose a better performing mask detector for
    > them.

-   Further improve and adopt the *COCO-annotator* for easy, fast and
    > cheap annotations. Put effort in tablet aspects and usage.

-   Further improve the *altered Augmentor*. Add HSV color alteration,
    > maybe incorporate the png-to-json script inside for seamless
    > augmentation.

-   Localize intersecting pictures in datasets and combine their
    > annotations

References
==========

Mask R-CNN paper -
[*https://arxiv.org/abs/1703.06870*](https://arxiv.org/abs/1703.06870)

Faster R-CNN paper -
[*https://arxiv.org/abs/1506.01497*](https://arxiv.org/abs/1506.01497)

Matterport Mask R-CNN -
[*https://github.com/matterport/Mask\_RCNN*](https://github.com/matterport/Mask_RCNN)

Dima repository of Mask R-CNN -

Original COCO-annotator -
[*https://github.com/jsbroks/coco-annotator*](https://github.com/jsbroks/coco-annotator)

Original Augmentor -
[*https://github.com/mdbloice/Augmentor*](https://github.com/mdbloice/Augmentor)

COCO dataset -
[*http://cocodataset.org/\#home*](http://cocodataset.org/#home)

COCO annotations from scratch -
[*http://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch*](http://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch)

Building a custom mask r-cnn -
[*https://towardsdatascience.com/building-a-custom-mask-rcnn-model-with-tensorflow-object-detection-952f5b0c7ab4*](https://towardsdatascience.com/building-a-custom-mask-rcnn-model-with-tensorflow-object-detection-952f5b0c7ab4)

.

**Synthetic stems\
**![](./media/image9.jpg){width="2.0833333333333335in"
height="1.375in"}![](./media/image31.jpg){width="2.0833333333333335in"
height="1.375in"}![](./media/image21.jpg){width="2.0833333333333335in"
height="1.375in"}

[^1]: See
    [*http://cocodataset.org/\#format-data*](http://cocodataset.org/#format-data)

[^2]: Original COCO-annotator by jsbroks -
    [*https://github.com/jsbroks/coco-annotator*](https://github.com/jsbroks/coco-annotator)

    Altered version for our project -
    [*https://github.com/simonlousky/cucu-annotator*](https://github.com/simonlousky/cucu-annotator)

[^3]: Original library -
    [*https://github.com/mdbloice/Augmentor*](https://github.com/mdbloice/Augmentor)\
    Altered version for our purposes -
    [*https://github.com/simonlousky/alteredAugmentor*](https://github.com/simonlousky/alteredAugmentor)

[^4]: The script is available in the altered COCO-Annotator repository

[^5]: Instructions on how to do that is found in the repository

[^6]: Instructions on how to use the Augmentor scripts can be found in
    the repository
