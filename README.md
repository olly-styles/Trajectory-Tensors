# Trajectory Tensors

This repo accompaines the following paper (CURRENTLY UNDER REVIEW):

Olly Styles, Tanaya Guha, Victor Sanchez, “Multi-Camera Trajectory Forecasting with Trajectory Tensors”, Subbmited to t-PAMI (UNDER REVIEW)

Preprint available: [[ArXiv link](https://arxiv.org/pdf/2108.04694.pdf)]

## Dataset download

All experiments are conducted on the [[WNMF dataset](https://github.com/olly-styles/Multi-Camera-Trajectory-Forecasting)], which is required to run the code.

## Installation

Clone repo:
```
git clone https://github.com/olly-styles/trajectory-tensors
```

Create virtual environment:
```
python3 -m venv venv
source venv/bin/activate
```

Install requirements:
```
pip install -r requirements.txt
```

## Preprocessing
After downloading the dataset, preprocess the data using the following command. Please note: This operation requires 85GB of storage. It will take a few minute to run, depending on your system:
```
python preprocessing/preprocess_all_data.py
```

After completing this operation, all tests should pass. You can test that the repo has installed correctly and the dataset has proccessed by running the tests:
```
python -m pytest testing
```

## Experiments
All experiments from the paper for the which, when, and where problem formulations are found in the ```experiments``` directory. Note that the CNN-GRU trajectory tensor model loads weights from the trajectory tensor autoencoder. There weights are created using ```experiments/autoencoder/autoencoder.py```

## Evaluation
After running the experiments and saving prediction, scripts to compute metrics are found in the ```evaluation``` directory.
