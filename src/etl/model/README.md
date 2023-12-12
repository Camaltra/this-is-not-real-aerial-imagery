# Classification (Ocean vs Earth) Model & Local Model Registry

# Table of Contents

1. [Overview](#overview)
2. [Purpose](#purpose)
3. [Modules](#modules)
    - [Trainer](#trainer)
    - [Tracker](#tracker)
    - [Experiments Logs Output](#experiments-logs-output)
    - [Available Models](#available-models)

## Overview

Before diving into the details, it's worth noting that this component has been designed with a level of complexity, providing a robust foundation for the project. Although it's a bit overengineered for the current scope, it offers room for improvement and showcases a binary classification system without a skewed dataset. The default metric used here is accuracy, and it holds a prominent position throughout the entire system.

The system is comprised of two major components hidden in the `./engine` folder:

1. The Trainer
2. The Tracker

These tools enable the creation of a customizable training environment through configurable variables, either through command line arguments or a Python configuration file.

## Purpose

The primary task is straightforward: differentiate between water and earth in aerial pictures. While a non-AI algorithm could suffice with good results, this project explores the implementation of famous model architectures for training and experimentation. Additionally, a custom experiment tracker has been created from scratch to assess its functionality.

## Modules

### Trainer

The trainer is a simple component that, given a model and some parameters, facilitates the training process. Currently, the BCTrainer (Binary Classification Trainer) is implemented, aligning with the project's purpose. However, the trainer is designed to be extendable to sub-child classes for additional tasks. The primary method is `fit()`.

### Tracker

The tracker is built to monitor experiments using a JSON database. All model training details are stored in JSON files, and model are also store along. The tracker keeps track of the current tags used and identifies the best model from the training process. It compiles information for a global experiment training overview as well as a more detailed version for each model. This interface involves the method `.record_training()`, and note that the tracker operates as a context manager.

### Experiments Logs Output

The tracker outputs experiment logs into a folder named `./registry`, containing all experiments classified by ID.

### Available Models

Find all available models for experiments tracking in the `deep_learning_models.py` file. The current available model is:

- Shallow Convnet

Additional models will be added over time. Note that this folder might eventually become its own repository to serve as a module. If this happens, a link will be provided here.

## Use
For now, no main script is provide, but it will been in some time

