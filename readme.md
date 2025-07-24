# DNet - Part of the Deep-Learning Model for Mirror Detection FYP

The source code of the DNet mirror detection network of Deep-Learning Model for Mirror Detection FYP (2025). 

## Prerequisites
- Linux or Windows
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started

### Using the Pretrained GeoNet Encoder
In order to train the model using the provided code, you should download the 7-type classification GeoNetM Encoder [here](https://drive.google.com/drive/folders/1k6kLYyvqaUST3m-odXj-Lxr6YPJ4TFo_?usp=sharing).

In this project, we use [MSD](https://drive.google.com/file/d/1Znw92fO6lCKfXejjSSyMyL1qtFepgjPI/view) and [PMD](https://drive.google.com/file/d/1xF_YLqbXRkB6JjXHgmM05mdPjlbtPmDz/view) datasets for training and evaluating our method.

### Pretrained DNet Models (MSD & PMD)
You can find the pretrained DNet models for [here](https://drive.google.com/drive/folders/1oUXMwZVaMkjMRUlqqzjaGyP2k59mwGuW?usp=sharing)

### Configuration
Before training and evaluation, you should specify the paths of the datasets, backbone, and pretrained encoder in config.py.

## How to Run
### Training
Run the following command for training:
```bash
python train.py
condor_submit [condor_script] # if you want to submit the task on HTCondor
```
You can modify the hyperparameters in the args variable.

### Testing
Run the following command for evaluation:
```bash
python infer.py
condor_submit [condor_script] # if you want to submit the task on HTCondor
```
