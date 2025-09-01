# Code for Conducting IDH Classification based on WSIs

Introduction
===
This repository provides the code for conducting IDH classification based on WSIs.
Before running the code, please make sure you have installed all the required packages. You can install them by running `pip install -r requirements.txt`.

Data Preprocessing
===
Before processing the slides, we suggest to arrange the slides in the following structure:
```
data/
    ├── source/
    │   ├── slide_name_1.svs
    │   ├── slide_name_2.svs
    │   └── ...
```
Data preprocessing was done following the steps in [CLAM](https://github.com/mahmoodlab/CLAM). This will generate the processed slide files in .h5 format which contains the patch coordinates.
An example of the processed slide file structure is as follows:
```
processed_slide/
    ├── source/
    |   |── patches/
    |   |   ├── slide_name_1.h5
    |   |   ├── slide_name_2.h5
    |   |   └── ...
```
Then, we will convert processed patches in to patch-level feature embeddings using a pre-trained model (default is ResNet50). The feature embeddings will be saved in .pt format. The resulting file structure is as follows:
```
processed_slide/
    ├── source/
    |   |── R50_features/
    |   |   ├── slide_name_1.pt
    |   |   ├── slide_name_2.pt
    |   |   └── ...
```

Training
====
We provide the training code for GPASS in `run.sh`. There are couple of parameters that can be modified in the `run.sh` file. 
DATA_NAME: the name of the dataset, e.g. ZJ, ZJ_test, etc.
MODEL_NAME: the name of the model, e.g. ResNet50, ResNet101, etc.
MODEL_NAME_MIL: the name of the MIL model, e.g. CLAM_SB, CLAM_GB, etc.

Training the model is as simple as running `./run.sh`. Running will start, and be default the model will be trained for 100 epochs. At the end of each epoch, the best-performing model will be saved to the `results` folder.

Testing
====
For testing the model, simply use the '--testing' flag in the `run.sh` file.

