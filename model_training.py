import numpy as np

import tensorflow as tf

from tensorflow.keras import models, layers
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from model_create import get_model
from data_processing import get_dirs

def get_data_ready():
    """
    Retrieves data paths and global parameters for training and testing.

    Returns:
        str: The path to the directory containing test images.

    Steps:
        1. Retrieve directory paths for labels, training/test images, and parameters.
        2. Load global parameters for training and augmentation.
        3. Return the path to the test image directory.
    """
    labels_dir, train_image_dir, test_image_dir, params_dir = get_dirs()
    global BATCH_SIZE, GAUSSIAN_NOISE, UPSAMPLE_MODE, NET_SCALING, IMG_SCALING, VALID_IMG_COUNT, MAX_TRAIN_STEPS, MAX_TRAIN_EPOCHS, AUGMENT_BRIGHTNESS, WEIGHT_DECAY
    
    BATCH_SIZE, GAUSSIAN_NOISE, UPSAMPLE_MODE, NET_SCALING, IMG_SCALING, VALID_IMG_COUNT, MAX_TRAIN_STEPS, MAX_TRAIN_EPOCHS, AUGMENT_BRIGHTNESS, WEIGHT_DECAY = get_params(params_dir)
    return test_image_dir

@tf.function
def dice_coef(y_true, y_pred):
    """
    Calculates the Dice coefficient between two tensors.

    Args:
        y_true (tf.Tensor): The ground truth tensor.
        y_pred (tf.Tensor): The predicted tensor.

    Returns:
        tf.Tensor: The Dice coefficient between the tensors.

    Steps:
        1. Cast both tensors to float32.
        2. Reshape both tensors to vectors.
        3. Calculate the intersection of the flattened tensors.
        4. Compute the Dice coefficient using the intersection and sum of both flattened tensors.
        5. Add a smoothing factor to avoid division by zero.
        6. Return the Dice coefficient.
    """
    smooth = 1.
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), dtype=tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), dtype=tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score
@tf.function
def dice_p_bce(in_gt, in_pred):
    """
    Calculates the negative of the Dice coefficient and combines it with the binary cross-entropy loss.

    Args:
        in_gt (tf.Tensor): The ground truth tensor.
        in_pred (tf.Tensor): The predicted tensor.

    Returns:
        tf.Tensor: The combined loss.
    """
    return - dice_coef(in_gt, in_pred)
@tf.function
def true_positive_rate(y_true, y_pred):
    """
    Calculates the true positive rate (TPR) metric.

    Args:
        y_true (tf.Tensor): The ground truth tensor.
        y_pred (tf.Tensor): The predicted tensor.

    Returns:
        tf.Tensor: The true positive rate.

    Steps:
        1. Flatten both tensors.
        2. Multiply the flattened tensors.
        3. Sum the product.
        4. Divide by the sum of the ground truth tensor.
        5. Return the true positive rate.
    """
    return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)

def fit(seg_model, train_df, valid_x, valid_y, aug_gen, callbacks_list):
    """
    Fits a segmentation model using an augmented data generator.

    Args:
        seg_model (Model): The segmentation model to be trained.
        train_df (pd.DataFrame): Pandas dataframe containing training data information.
        valid_x (np.ndarray): Validation images.
        valid_y (np.ndarray): Validation masks.
        aug_gen (generator): Augmented data generator.
        callbacks_list (list): List of callbacks to be used during training.

    Returns:
        list: List of training history objects from each epoch.

    Steps:
        1. Compile the model with Adam optimizer, dice_p_bce loss, and dice_coef and true_positive_rate metrics.
        2. Calculate the maximum number of training steps based on batch size and data size.
        3. Initialize an empty list to store training history objects.
        4. Train the model using the provided augmented data generator, callbacks, and validation data.
        5. Append the training history object for each epoch to the list.
        6. Return the list of training history objects.
    """
    seg_model.compile(optimizer=Adam(learning_rate=1e-3), loss=dice_p_bce, metrics=[dice_coef, true_positive_rate])
    
    step_count = min(MAX_TRAIN_STEPS, train_df.shape[0]//BATCH_SIZE)
    loss_history = [seg_model.fit_generator(aug_gen,
                                 steps_per_epoch=step_count,
                                 epochs=MAX_TRAIN_EPOCHS,
                                 validation_data=(valid_x, valid_y),
                                 callbacks=callbacks_list,
                                workers=1
                                           )]
    return loss_history

def train_model(to_train):
    """
    Trains a segmentation model.

    Args:
        to_train (bool): Whether to train the model or load existing weights. Defaults to True.

    Returns:
        Model: The trained segmentation model.

    Steps:
        1. Get the augmented data generator, validation data, and segmentation model.
        2. Define checkpoint, learning rate reduction, and early stopping callbacks.
        3. If training is requested:
            * Enter a loop until the validation loss falls below a threshold.
            * Train the model using the specified callbacks.
        4. Return the trained segmentation model.
    """
    aug_gen, valid_x, valid_y, seg_model, train_df = get_model()
    
    weight_path="results/{}_weights.best.hdf5".format('seg_model')

    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)

    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.25,
                                    patience=3, verbose=1, mode='min',
                                    min_delta=0.0001, cooldown=0, min_lr=1e-8)

    early = EarlyStopping(monitor="val_loss", mode="min", verbose=2,
                        patience=20)

    callbacks_list = [checkpoint, early, reduceLROnPlat]
    if to_train:
        while True:
            loss_history = fit(seg_model, train_df, valid_x, valid_y, aug_gen, callbacks_list)
            if np.min([mh.history['val_loss'] for mh in loss_history]) < 0.01:
                break
    return seg_model
        
def train_full(to_train = True):
    """
    Trains a full-resolution segmentation model.

    Args:
        to_train (bool, optional): Whether to train the model or load existing weights. Defaults to True.

    Returns:
        Model: The trained full-resolution segmentation model.

    Steps:
        1. Train a model using the `train_model` function.
        2. Load the model weights from the best saved checkpoint.
        3. Save the model as `results/seg_model.h5`.
        4. If image scaling is applied, create a full-resolution model as follows:
            * Add an `AvgPool2D` layer with the scaling factor as input.
            * Add the trained segmentation model.
            * Add an `UpSampling2D` layer with the scaling factor.
        5. Otherwise, use the trained model as the full-resolution model.
        6. Save the full-resolution model as `results/fullres_model.h5`.
        7. Return the full-resolution model.
    """

    seg_model = train_model(to_train)
    weight_path="results/{}_weights.best.hdf5".format('seg_model')
    seg_model.load_weights(weight_path)
    seg_model.save('results/seg_model.h5')
    
    if IMG_SCALING is not None:
        fullres_model = models.Sequential()
        fullres_model.add(layers.AvgPool2D(IMG_SCALING, input_shape = (None, None, 3)))
        fullres_model.add(seg_model)
        fullres_model.add(layers.UpSampling2D(IMG_SCALING))
    else:
        fullres_model = seg_model
    fullres_model.save('results/fullres_model.h5')
    return fullres_model
