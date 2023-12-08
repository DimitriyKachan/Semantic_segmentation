import tensorflow as tf

from tensorflow.keras import models, layers
from data_processing import get_dirs, get_params, data_prep, data_ready



def get_data_ready():
    """
    Retrieves data paths, global parameters for training and augmentation, and prepares augmented data and validation data.

    Returns:
        tuple: A tuple containing the following elements:
            * aug_gen: A generator yielding augmented image and label pairs for training.
            * valid_x: A numpy array of validation images.
            * valid_y: A numpy array of validation labels.
    """
    labels_dir, train_image_dir, test_image_dir, params_dir = get_dirs()
    global BATCH_SIZE, GAUSSIAN_NOISE, UPSAMPLE_MODE, NET_SCALING, IMG_SCALING, VALID_IMG_COUNT, MAX_TRAIN_STEPS, MAX_TRAIN_EPOCHS, AUGMENT_BRIGHTNESS, WEIGHT_DECAY
        
    BATCH_SIZE, GAUSSIAN_NOISE, UPSAMPLE_MODE, NET_SCALING, IMG_SCALING, VALID_IMG_COUNT, MAX_TRAIN_STEPS, MAX_TRAIN_EPOCHS, AUGMENT_BRIGHTNESS, WEIGHT_DECAY = get_params(params_dir)
    
    aug_gen, valid_x, valid_y, t_x, train_df = data_prep(labels_dir, train_image_dir)
    
    return aug_gen, valid_x, valid_y

def build_model(t_x):
    """
    Builds the segmentation model using Keras functional API.

    Args:
        t_x (np.ndarray): An example image used to infer the input shape for the model.

    Returns:
        Model: The compiled segmentation model.

    Steps:
        1. Define custom upsampling functions based on the chosen mode (`DECONV` or `SIMPLE`).
        2. Define the input layer and optionally apply downsampling if specified.
        3. Add Gaussian noise and batch normalization to the input.
        4. Define convolutional layers with Leaky ReLU activation and L2 weight regularization.
        5. Use max pooling for downsampling and batch normalization after each pooling layer.
        6. Use upsampling and concatenation to merge features from different levels.
        7. Add final convolutional layers with ReLU activation and L2 weight regularization.
        8. Apply a 1x1 convolution with sigmoid activation as the output layer.
        9. Optionally upscale the output if downsampling was applied to the input.
        10. Create and return the Keras model.
    """
    def upsample_conv(filters, kernel_size, strides, padding):
        return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)
    def upsample_simple(filters, kernel_size, strides, padding):
        return layers.UpSampling2D(strides)

    if UPSAMPLE_MODE=='DECONV':
        upsample=upsample_conv
    else:
        upsample=upsample_simple
        
    input_img = layers.Input(t_x.shape[1:], name = 'RGB_Input')
    pp_in_layer = input_img
    if NET_SCALING is not None:
        pp_in_layer = layers.AvgPool2D(NET_SCALING)(pp_in_layer)
        
    pp_in_layer = layers.GaussianNoise(GAUSSIAN_NOISE)(pp_in_layer)
    pp_in_layer = layers.BatchNormalization()(pp_in_layer)

    c1 = layers.Conv2D(32, (3, 3), activation='leaky_relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)) (pp_in_layer)
    c1 = layers.Conv2D(32, (3, 3), activation='leaky_relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)) (c1)
    p1 = layers.MaxPooling2D((2, 2)) (c1)
    p1 = layers.BatchNormalization()(p1)

    c2 = layers.Conv2D(64, (3, 3), activation='leaky_relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)) (p1)
    c2 = layers.Conv2D(64, (3, 3), activation='leaky_relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)) (c2)
    p2 = layers.MaxPooling2D((2, 2)) (c2)
    p2 = layers.BatchNormalization()(p2)

    c3 = layers.Conv2D(128, (3, 3), activation='leaky_relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)) (p2)
    c3 = layers.Conv2D(128, (3, 3), activation='leaky_relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)) (c3)
    p3 = layers.MaxPooling2D((2, 2)) (c3)
    p3 = layers.BatchNormalization()(p3)

    c4 = layers.Conv2D(256, (3, 3), activation='leaky_relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)) (p3)
    c4 = layers.Conv2D(256, (3, 3), activation='leaky_relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)) (c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = layers.BatchNormalization()(p4)


    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)) (p4)
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)) (c5)

    u6 = upsample(256, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)) (u6)
    c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)) (c6)

    u7 = upsample(128, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)) (u7)
    c7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)) (c7)

    u8 = upsample(64, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)) (u8)
    c8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)) (c8)

    u9 = upsample(32, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)) (u9)
    c9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)) (c9)

    d = layers.Conv2D(1, (1, 1), activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)) (c9)
    if NET_SCALING is not None:
        d = layers.UpSampling2D(NET_SCALING)(d)
    seg_model = models.Model(inputs=[input_img], outputs=[d])
    
    return seg_model

def get_model():
    """
    Prepares data and builds the segmentation model.

    Returns:
        tuple: A tuple containing the following elements:
            * aug_gen: A generator yielding augmented image and label pairs for training.
            * valid_x: A numpy array of validation images.
            * valid_y: A numpy array of validation labels.
            * seg_model: The compiled segmentation model.
            * train_df: A Pandas dataframe containing training image information.

    Steps:
        1. Use `data_ready` function to prepare augmented data and validation data.
        2. Call `build_model` function to build the segmentation model.
        3. Return a tuple containing the prepared data and the model.
    """
    aug_gen, valid_x, valid_y, t_x, train_df = data_ready()
    seg_model = build_model(t_x)
    return aug_gen, valid_x, valid_y, seg_model, train_df