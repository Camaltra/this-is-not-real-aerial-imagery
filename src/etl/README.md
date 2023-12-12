# ETL Process for Google Earth Data Collection

## Table of Contents

1. [Overview](#overview)
   - [Working](#working)
   
2. [How to Use](#how-to-use)
   - [Required Steps](#required-steps)
   - [Parameters](#parameters)
   - [Running the Script](#running-the-script)

3. [Data Storage](#data-storage)

## Overview

The ETL (Extract, Transform, Load) process described here is designed to gather data for training the main model, specifically the Diffusion model. Rather than downloading the Google Earth application, this solution involves interacting with the web-based version.

### Working

The script outlined below performs the following actions from an open Google Earth window:

- Captures a screenshot of a specified size
- Determines whether to keep the screenshot based on the use of an AI model
- Moves to an unknown tile using keyboard inputs
- Repeats the process
- Saves a batch of images

This approach allows for the systematic gathering of data from various edges.

## How to Use

1. Open your web browser, navigate to Google Earth, and manually select your starting point.
2. Launch the script with the following parameters:

   - `--number-batch`: Number of batches desired
   - `--batch-size`: Number of captures per batch
   - `--image-collection-offset`: Displacement value (number of times arrow keys will be pressed to move in a given direction)
   - `--screenshot-width`: Image width
   - `--screenshot-height`: Image height
   - `--use-classifier`: Whether to use a classifier (refer to the model README for Classifier Earth vs Ocean)
   - `--classifier-model-tag`: Tag of the classifier to use (see model README)
   - `--delete-intermediate-saves`: Delete intermediate saves

   Note: The only required parameter is the number of batches; all other parameters have default values. Refer to the script's `parse_arg` function for details.

3. Run the script using the following command:

   ```bash
   python main.py <parameters>
   ```

The data will be stored in:

Intermediate data: ./temporary_data_save
Final pipeline output: ../data
All data will be saved with the .npy file extension.

------
Feel free to adjust any part of this according to your preferences or specific details you'd like to emphasize.