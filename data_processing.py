import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import cv2
import os
import json

from skimage.util import montage
from skimage.morphology import label

from tensorflow.keras.preprocessing.image import ImageDataGenerator

montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)

def get_dirs():
    """Function for getting the directories

    Returns:
        str: path to csv file containing labels
        str: path to train images
        str: path to test images
        str: path to json file containing paths
    """
    paths = json.load(open('paths.json'))

    labels_dir =paths["labels_dir"]
    train_image_dir =paths["train_image_dir"]
    test_image_dir =paths["test_image_dir"]

    params_dir = paths["params"]
    
    return labels_dir, train_image_dir, test_image_dir, params_dir

def get_params(params_dir):
    """Function to get global parameters

    Args:
        params_dir (str): path to json file containing paths

    Returns:
        int: amount of images in a single batch
        float: standard deviation of gaussian noise
        str: upsample mode
        tuple: scale of network
        tuple: scale of images
        int: amount of validation images
        int: maximum number of training steps
        int: maximum number of training epochs
        bool: whether to augment brightness
        float: weight decay
    """
    params = json.load(open(params_dir))
    
    global BATCH_SIZE, GAUSSIAN_NOISE, UPSAMPLE_MODE, NET_SCALING, IMG_SCALING, VALID_IMG_COUNT, MAX_TRAIN_STEPS, MAX_TRAIN_EPOCHS, AUGMENT_BRIGHTNESS, WEIGHT_DECAY
    
    BATCH_SIZE = params["BATCH_SIZE"]
    GAUSSIAN_NOISE = params["GAUSSIAN_NOISE"]
    UPSAMPLE_MODE = params["UPSAMPLE_MODE"]
    NET_SCALING = (params["NET_SCALING_val"], params["NET_SCALING_val"])
    IMG_SCALING = (params["IMG_SCALING_val"], params["IMG_SCALING_val"])
    VALID_IMG_COUNT = params["VALID_IMG_COUNT"]
    MAX_TRAIN_STEPS = params["MAX_TRAIN_STEPS"]
    MAX_TRAIN_EPOCHS = params["MAX_TRAIN_EPOCHS"]
    AUGMENT_BRIGHTNESS = bool(params["AUGMENT_BRIGHTNESS"])
    WEIGHT_DECAY = params["WEIGHT_DECAY"]
    
    return BATCH_SIZE, GAUSSIAN_NOISE, UPSAMPLE_MODE, NET_SCALING, IMG_SCALING, VALID_IMG_COUNT, MAX_TRAIN_STEPS, MAX_TRAIN_EPOCHS, AUGMENT_BRIGHTNESS, WEIGHT_DECAY

def mask_decoder(lable, shape = (768, 768)):
    """Function to decode a strinf into a mask

    Args:
        lable (str): str representation of mask
        shape (tuple, optional): size of an image. Defaults to (768, 768).

    Returns:
        ndarray: mask where ones are ship borders and rest is zero
    """
    splited = lable.split()
    starts, lengths = [np.asarray(x, dtype=np.longlong) for x in (splited[0:][::2], splited[1:][::2])]
    starts -=1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for low, high in zip(starts, ends):
        img[low:high] = 1
    return img.reshape(shape).T

def combine_masks(mask_list):
    """
    Combines a list of segmentation masks into a single mask.

    Args:
        mask_list (list): A list of segmentation masks. Each mask can be:
            * A numpy array of shape (768, 768) containing binary values (0 or 1).
            * A string representing the encoded mask in the RLE format.

    Returns:
        np.ndarray: A 3D numpy array of shape (768, 768, 1) containing the combined mask.
            Values are 0 or 1, where 1 indicates the presence of a pixel belonging to any mask in the list.
    """
    masks = np.zeros((768,768), dtype=np.int32)
    for mask in mask_list:
        if isinstance(mask, str):
            masks += mask_decoder(mask)
    return np.expand_dims(masks, -1)

def mask_encoder(img):
    """
    Encodes a segmentation mask into RLE format.

    Args:
        img (np.ndarray): The 2D segmentation mask of shape (height, width) with values 0 or 1.

    Returns:
        str: The encoded mask in RLE format.
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:]!=pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def multi_mask_encoder(img):
    """
    Encodes multiple segmentation masks within a single image into RLE format.

    Args:
        img (np.ndarray): The 3D image with one channel containing multiple masks. Each mask has a unique integer value.

    Returns:
        list[str]: A list of encoded masks in RLE format, one for each unique mask found in the image.
    """

    labels = label(img[:, :, 0])
    return [mask_encoder(labels==k) for k in np.unique(labels[labels>0])]

def sample_ships(in_df, base_rep_val=1500):
    """
    Samples images from the input dataframe, ensuring balanced representation of ship count.

    Args:
        in_df (pd.DataFrame): The dataframe containing image information, including ship count.
        base_rep_val (int, optional): The base number of repetitions for each ship count. Defaults to 1500.

    Returns:
        pd.DataFrame: A subsampled dataframe with balanced ship count representation.
    """
    if in_df['ships'].values[0]==0:
        return in_df.sample(base_rep_val//2)
    elif in_df['ships'].values[0]>8:
        return in_df.sample(base_rep_val//2)
    else:
        return in_df.sample(base_rep_val, replace=(in_df.shape[0]<base_rep_val))

def data_prep(labels_dir, train_image_dir):
    """
    Prepares and augments data for training a ship detection model.

    Args:
        labels_dir (str): Path to the CSV file containing image labels.
        train_image_dir (str): Path to the directory containing training images.

    Returns:
        tuple: A tuple containing the following elements:
            * aug_gen: A generator yielding augmented image and label pairs for training.
            * valid_x: A numpy array of validation images.
            * valid_y: A numpy array of validation labels.
            * t_x: A numpy array of augmented training images.
            * train_df: A Pandas dataframe containing training image information.

    Steps:
        1. Read the labels CSV file and preprocess data.
        2. Split data into training and validation sets.
        3. Group training images by ship count and sample balanced batches.
        4. Define data augmentation parameters.
        5. Create image and label augmentation generators.
        6. Generate augmented training and validation data.
        7. Return augmented data and training information.
    """
    lables = pd.read_csv(labels_dir)
    lables['ships'] = lables['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
    unique_img_ids = lables.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
    unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)
    unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])

    unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(lambda c_img_id:os.stat(os.path.join(train_image_dir,c_img_id)).st_size/1024)

    unique_img_ids = unique_img_ids[unique_img_ids['file_size_kb']>50] # keep only 50kb files
    lables.drop(['ships'], axis=1, inplace=True)
    
    train_ids, valid_ids = train_test_split(unique_img_ids, test_size = 0.3, stratify = unique_img_ids['ships'])
    train_df = pd.merge(lables, train_ids)
    valid_df = pd.merge(lables, valid_ids)
    
    train_df['grouped_ship_count'] = train_df['ships'].map(lambda x: (x+1)//2).clip(0, 7)
    
    valid_x, valid_y = next(batch_image_gen(valid_df, train_image_dir, VALID_IMG_COUNT))
    
    dg_args = dict(featurewise_center = False, 
                  samplewise_center = False,
                  rotation_range = 45, 
                  width_shift_range = 0.1, 
                  height_shift_range = 0.1, 
                  shear_range = 0.01,
                  zoom_range = [0.9, 1.25],  
                  horizontal_flip = True, 
                  vertical_flip = True,
                  fill_mode = 'reflect',
                   data_format = 'channels_last')
    
    if AUGMENT_BRIGHTNESS:
        dg_args[' brightness_range'] = [0.5, 1.5]
    image_gen = ImageDataGenerator(**dg_args)

    if AUGMENT_BRIGHTNESS:
        dg_args.pop('brightness_range')
    label_gen = ImageDataGenerator(**dg_args)

    def create_aug_gen(in_gen, seed = None):
        """
        Creates an augmented image and label generator based on an input generator.

        Args:
            in_gen (generator): The input generator yielding pairs of images and labels.
            seed (int, optional): The random seed to use for augmentation. Defaults to None.

        Yields:
            tuple: A tuple of augmented images and labels.
        """
        np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
        for in_x, in_y in in_gen:
            seed = np.random.choice(range(9999))
            g_x = image_gen.flow(255*in_x, 
                                batch_size = in_x.shape[0], 
                                seed = seed, 
                                shuffle=True)
            g_y = label_gen.flow(in_y, 
                                batch_size = in_x.shape[0], 
                                seed = seed, 
                                shuffle=True)

            yield next(g_x)/255.0, next(g_y)

    aug_gen = create_aug_gen(batch_image_gen(train_df))
    
    train_df['grouped_ship_count'] = train_df['ships'].map(lambda x: (x+1)//2).clip(0, 7)
    
    balanced_train_df = train_df.groupby('grouped_ship_count').apply(sample_ships)
    
    train_gen = batch_image_gen(balanced_train_df)
    cur_gen = create_aug_gen(train_gen)
    t_x, t_y = next(cur_gen)
    
    return aug_gen, valid_x, valid_y, t_x, train_df

def batch_image_gen(in_df, train_image_dir, batch_size=BATCH_SIZE):
    """
    Generates batches of images and masks from a pandas dataframe.

    Args:
        in_df (pd.DataFrame): The dataframe containing image information and masks.
        train_image_dir (str): Path to the directory containing training images.
        batch_size (int, optional): The number of images per batch. Defaults to BATCH_SIZE.

    Yields:
        tuple: A tuple containing the following elements:
            * rgb: A numpy array of images with shape (batch_size, height, width, 3).
            * mask: A numpy array of masks with shape (batch_size, height, width, 1).

    Steps:
        1. Group images by their ImageId.
        2. Initialize empty lists for RGB images and masks.
        3. Enter an infinite loop to generate batches.
        4. Shuffle the list of image groups.
        5. For each image group:
            * Read the RGB image and encoded masks.
            * Combine the masks into a single mask.
            * Apply any image scaling if specified.
            * Add the RGB image and mask to their respective lists.
            * Check if enough images are collected for a batch.
                * If so, yield the batch as numpy arrays.
                * Otherwise, continue adding images to the lists.
    """
    all_batches = list(in_df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(train_image_dir, c_img_id)
            c_img = cv2.imread(rgb_path)
            c_mask = combine_masks(c_masks['EncodedPixels'].values)
            if IMG_SCALING is not None:
                c_img = c_img[::IMG_SCALING[0], ::IMG_SCALING[1]]
                c_mask = c_mask[::IMG_SCALING[0], ::IMG_SCALING[1]]
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb)>=batch_size:
                yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)
                out_rgb, out_mask=[], []

def data_ready():
    """
    Prepares data for model training and validation.

    Returns:
        tuple: A tuple containing the following elements:
            * aug_gen: A generator yielding augmented image and label pairs for training.
            * valid_x: A numpy array of validation images.
            * valid_y: A numpy array of validation labels.

    Steps:
        1. Retrieve directory paths for labels, training/test images, and parameters.
        2. Load model parameters from the specified directory.
        3. Prepare and augment training data.
        4. Return augmented training data and validation data.
    """
    labels_dir, train_image_dir, test_image_dir, params_dir = get_dirs()
    get_params(params_dir)
    
    aug_gen, valid_x, valid_y, t_x, train_df = data_prep(labels_dir, train_image_dir)
    
    return aug_gen, valid_x, valid_y