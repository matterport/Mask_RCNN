# Nuclei Counting and Segmentation

This sample implements the [2018 Data Science Bowl challenge](https://www.kaggle.com/c/data-science-bowl-2018).
The goal is to segment individual nuclei in microscopy images.
The `nucleus.py` file contains the main parts of the code, and the two Jupyter notebooks


## Command line Usage
Train a new model starting from ImageNet weights using `train` dataset (which is `stage1_train` minus validation set)
```
python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=imagenet
```

Train a new model starting from specific weights file using the full `stage1_train` dataset
```
python3 nucleus.py train --dataset=/path/to/dataset --subset=stage1_train --weights=/path/to/weights.h5
```

Resume training a model that you had trained earlier
```
python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=last
```

Generate submission file from `stage1_test` images
```
python3 nucleus.py detect --dataset=/path/to/dataset --subset=stage1_test --weights=<last or /path/to/weights.h5>
```


## Jupyter notebooks
Two Jupyter notebooks are provided as well: `inspect_nucleus_data.ipynb` and `inspect_nucleus_model.ipynb`.
They explore the dataset, run stats on it, and go through the detection process step by step.
