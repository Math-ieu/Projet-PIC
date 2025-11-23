import tensorflow as tf

def augment_image(image, seed=None):
    """
    Apply data augmentation to an image.
    """
    # Random flip
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
    
    # Random brightness/contrast
    image = tf.image.random_brightness(image, max_delta=0.1, seed=seed)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1, seed=seed)
    
    # Random saturation/hue (only for RGB)
    if image.shape[-1] == 3:
        image = tf.image.random_saturation(image, lower=0.9, upper=1.1, seed=seed)
        image = tf.image.random_hue(image, max_delta=0.02, seed=seed)
        
    return image

def get_augmentation_layer():
    """
    Returns a Keras Sequential model for augmentation (can be part of the model).
    """
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])
    return data_augmentation
