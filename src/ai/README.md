# Diffusion Model training

## Table of Contents

1. [Overview](#overview)
   - [Working](#working)
   - [Installation](#installation)
   
2. [How to Use](#how-to-use)

## Overview

This subfolder contain all the model and the training configuration for the Diffusion Model. Majors components are models itself (`./models/u_net.py`) but also the training script (Fully parametrable) and the sampling script (Fully parametrable)

### Working

The diffusion process will be explained in a futur blog, hosted on medium. Short to know, the model, once trained, will create sample that have never been in the training set. It so, create deep fake. There is not correlation between the whole project and the model and training part, if you need to change the data to fit to our project, feel free to.

### Installation
Please install the envirement conda present in the `environment.yaml` file

## How to Use

The easiest way to use it is to fill the `training_model_config.json` with the config you need to use. Then you can just start the training
```bash
python train.py -c trained_model_config_128.json
```
You can also retrive a model semi train to finish the training 
```bash
python train.py -c trained_model_config_128.json -m 37
```
All the saved model and saved sample will be output in the `./results/oneres_128_1`

To create samples, please use the `sample.py` file and give the same configuration you have for the training. Also specify which model you want to use.
```bash
python sample.py -c trained_model_config_128.json -m 37 -n 6
```

WARNING: Don't use the -o parameters, it's only here cause some of my model have been trained on a differents model (That buffer some unwanted value) and need to delete them before load the model, for you it's only somethign you'll never use

All model have been tested and work on data collected through ETL process. Different levels of complexity are added between model, depending on how much the model need to be complexe for your task.

For my purpose the simpliest one work like a charm.