![CrowdAI-Logo](https://github.com/crowdAI/crowdai/raw/master/app/assets/images/misc/crowdai-logo-smile.svg?sanitize=true)

# crowdAI Mapping Challenge : Baseline

This repository contains the details of implementation of the Baseline submission using [Mask RCNN](https://arxiv.org/abs/1703.06870) which obtains a score of `[AP(IoU=0.5)=0.697 ; AR(IoU=0.5)=0.479]` for the [crowdAI Mapping Challenge](https://www.crowdai.org/challenges/mapping-challenge).

# Installation
```
git clone https://github.com/spMohanty/crowdai-mapping-challenge-mask-rcnn
cd crowdai-mapping-challenge-mask-rcnn
# Please ensure that you use python3.6
pip install -r requirements.txt
python setup.py install
```

# Notebooks
Please follow the instructions on the relevant notebooks for the training, prediction and submissions.

* [Training](Training.ipynb)
* [Prediction and Submission](Prediction and Submission.ipynb)
  (_pre-trained weights for baseline submission included_)

# Results
![sample_predictions](images/predictions.png)

# Citation
```
@misc{crowdAIMappingChallengeBaseline2018,
  author = {Mohanty, Sharada Prasanna},
  title = {CrowdAI Mapping Challenge 2018 : Baseline with Mask RCNN},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/spMohanty/crowdai-mapping-challenge-mask-rcnn}},
  commit = {a83c04bf906c2e286cddb2911e2b96d0e7d56bac}
}
```
# Acknowledgements
This repository heavily reuses code from the amazing [tensorflow Mask RCNN implementation](https://github.com/matterport/Mask_RCNN) by [@waleedka](https://github.com/waleedka/).
Many thanks to all the contributors of that project.

# Author
Sharada Mohanty [sharada.mohanty@epfl.ch](sharada.mohanty@epfl.ch)
