import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from data_processing  import get_dirs

from model_training import train_full

def main():
    """
    Performs the main workflow of the program:

    1. Retrieves directory paths for labels, training/test images, and parameters.
    2. Trains a full-resolution model.
    3. Loads the trained model weights.
    4. Predicts segmentation masks for test images.
    5. Visualizes the results.

    Steps:
        1. Retrieve directory paths.
        2. Train and load full-resolution model.
        3. Get list of test image paths.
        4. Create a figure and subplots for visualization.
        5. Loop through test images:
            * Read the image.
            * Expand the image to a batch of size 1.
            * Predict the segmentation mask using the loaded model.
            * Display the original image and predicted mask in subplots.
        6. Save the visualization figure.
    """
    labels_dir, train_image_dir, test_image_dir, params_dir = get_dirs()
    
    fullres_model = train_full(False)
    fullres_model.load_weights('results/fullres_model.h5')
    
    test_paths = os.listdir(test_image_dir)
    fig, m_axs = plt.subplots(20, 2, figsize = (10, 40))
    [c_ax.axis('off') for c_ax in m_axs.flatten()]
    for (ax1, ax2), c_img_name in zip(m_axs, test_paths):
        c_path = os.path.join(test_image_dir, c_img_name)
        c_img = cv2.imread(c_path)
        first_img = np.expand_dims(c_img, 0)/255.0
        first_seg = fullres_model.predict(first_img)
        ax1.imshow(first_img[0])
        ax1.set_title('Image')
        ax2.imshow(first_seg[0, :, :, 0], vmin = 0, vmax = 1)
        ax2.set_title('Prediction')
    fig.savefig('results/test_predictions.png')