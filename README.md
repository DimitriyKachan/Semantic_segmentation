# Semantic_segmentation
Semantic segmentation using Unet architecture
<br/><br/>
# Project Description

This project aims to detect and segment ships in satellite images using the U-Net deep learning model. It provides accurate and efficient ship detection, assisting various maritime applications such as navigation, traffic monitoring, and environmental protection.
<br/><br/>
# Features

Implements the U-Net architecture specifically designed for image segmentation tasks.

Achieves high accuracy in ship detection on the Airbus Ship Detection Challenge dataset.

Offers flexibility to adjust hyperparameters through a dedicated parameters.json file.

Includes various files for different functionalities:

-eda.ipynb: Exploratory data analysis of the dataset.

-data_processing.py: Data processing and augmentation pipeline.

-model_create.py: U-Net model architecture definition.

-model_training.py: Training functionality with customizable parameters.

-paths.json: Defines paths to dataset and results.

-main.ipynb: Main notebook showcasing the full workflow.

Saves trained model weights for later use or evaluation.

Provides example images of segmentation and prediction results.
<br/><br/>
# Installation and Usage

Clone this repository onto your local machine.

Create a folder named data within the repository directory.

Download and place the Airbus Ship Detection Challenge dataset inside the data folder.

Ensure you have Python 3 and necessary libraries installed. You can either install them according to requirements.txt, or import it through anaconda backup called UNET_env.yaml

Run the main.ipynb notebook for a complete demonstration of the entire workflow.

<br/><br/>

Alternatively, utilize individual script files for specific tasks:

-eda.ipynb: Explore and analyze the dataset.
-data_processing.py: Preprocess and augment data before training.
-model_create.py: Define and build the U-Net architecture.
-model_training.py: Train the model with adjustable parameters.
