import tensorflow as tf

def normalize_image(image):
    """
    Normalize image to [0, 1].
    """
    return tf.cast(image, tf.float32) / 255.0

def denormalize_image(image):
    """
    Convert back to [0, 255] uint8.
    """
    return tf.cast(image * 255.0, tf.uint8)

def resize_image(image, size=(256, 256)):
    """
    Resize image.
    """
    return tf.image.resize(image, size)

def prepare_input(image, target_size=(256, 256)):
    """
    Full preparation pipeline: resize -> normalize.
    """
    image = resize_image(image, target_size)
    image = normalize_image(image)
    return image
