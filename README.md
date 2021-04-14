![CrowdAI-Logo](https://github.com/crowdAI/crowdai/raw/master/app/assets/images/misc/crowdai-logo-smile.svg?sanitize=true)

>The research paper summarizing the corresponding benchmark and associated solutions can be found here : [Deep Learning for Understanding Satellite Imagery: An Experimental Survey](https://www.frontiersin.org/articles/10.3389/frai.2020.534696/full)

# crowdAI Mapping Challenge : Baseline

This repository contains the details of implementation of the Baseline submission using [Mask RCNN](https://arxiv.org/abs/1703.06870) which obtains a score of `[AP(IoU=0.5)=0.697 ; AR(IoU=0.5)=0.479]` for the [crowdAI Mapping Challenge](https://www.crowdai.org/challenges/mapping-challenge).

# Installation
```
git clone https://github.com/crowdai/crowdai-mapping-challenge-mask-rcnn
cd crowdai-mapping-challenge-mask-rcnn
# Please ensure that you use python3.6
pip install -r requirements.txt
python setup.py install
```

# Notebooks
Please follow the instructions on the relevant notebooks for the training, prediction and submissions.

* [Training](Training.ipynb)
* [Prediction and Submission](Prediction-and-Submission.ipynb)
  (_pre-trained weights for baseline submission included_)

# Results
![sample_predictions](images/predictions.png)

# Citation
```
@article{mohanty2020deep, 
    title={Deep Learning for Understanding Satellite Imagery: An Experimental Survey}, 
    author={Mohanty, Sharada Prasanna and Czakon, Jakub and Kaczmarek, Kamil A and Pyskir, Andrzej and Tarasiewicz, Piotr and Kunwar, Saket and Rohrbach, Janick and Luo, Dave and Prasad, Manjunath and Fleer, Sascha and others}, 
    journal={Frontiers in Artificial Intelligence}, 
    volume={3}, 
    year={2020}, 
    publisher={Frontiers Media SA}
}

@misc{crowdAIMappingChallengeBaseline2018,
    author = {Mohanty, Sharada Prasanna},
    title = {CrowdAI Mapping Challenge 2018 : Baseline with Mask RCNN},
    year = {2018},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/crowdai/crowdai-mapping-challenge-mask-rcnn}},
    commit = {bac1cf19adbc9d078122c6933da6f808c4ee590d}
}
```
# Acknowledgements
This repository heavily reuses code from the amazing [tensorflow Mask RCNN implementation](https://github.com/matterport/Mask_RCNN) by [@waleedka](https://github.com/waleedka/).
Many thanks to all the contributors of that project.
You are encouraged to checkout [https://github.com/matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN) for documentation on many other aspects of this code.

# Author
Sharada Mohanty [sharada.mohanty@epfl.ch](sharada.mohanty@epfl.ch)
