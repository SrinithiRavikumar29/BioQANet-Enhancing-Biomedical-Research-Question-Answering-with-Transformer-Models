# PubmedQA system

This repo contains the code for BioQANet: Enhancing Biomedical Research Question Answering with Transformer Models

Team Members:

- Srinithi Ravikumar
- Rithic Kumar Nandakumar

## Repo structure

### Folders
- data : contains the data required for experiments. Create the folder and place data from original PubMedQA here
- preprocess : contains the original script from PubmedQA for splitting expert into train and test
- predictions : contains the predicted json files
- Archived notebooks : Self explanatory

### Notebooks

- EDA.ipynb : Basic exploratory analysis and simple baselines for QA model
- Contrastive-512.ipynb : Notebook for training and inference of BioQANet
- Contrastive-without-pretraining.ipynb : Notebook for training and inference without pretraining
- Contrastive.ipynb : Similar to Contrastive-512 but has max length 400

### Scripts

- utils.py : Utility script
- contrastive-utils.py: Functions for model definition used in Contrastive-512 notebook
- evaluation.py: Original function from PubmedQA for evaluating the predictions
- get_human_performance.py: Original function from PubmedQA for getting human performance metrics

## Evaluation

python evaluation.py path_to_pred.json