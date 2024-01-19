# Unsupervised Maze Solver

## Overview
The Unsupervised Maze Solver is a working-progress project that uses a convolutional neural network (CNN) to autonomously learn and predict the exit path in maze images without the need for explicit ground truth labels.

## Project Structure
- src/: Contains source code files
  - model.py: Defines the Unsupervised Maze Solver CNN.
  - train_unsupervised.py: Script for training the unsupervised model.
  - infer_unsupervised.py: Script for using the trained model for inference.
- data/: Placeholder for your maze dataset
- saved_models/: Location to save your trained models
- requirements.txt: List of project dependencies
- README.md: Project documentation

## Getting Started
1. Install dependencies using pip install -r requirements.txt.
2. Prepare your maze dataset and place it in the data/ folder.
3. Train the unsupervised model by running python src/train_unsupervised.py.
4. Use the trained model for inference with python src/infer_unsupervised.py.

## Model Customization
Feel free to adjust the model architecture, hyperparameters, or loss functions based on your specific maze-solving requirements. Experiment with different configurations to achieve optimal performance.
