import tensorflow as tf
from tensorflow.keras import layers, models

def create_autoencoder(input_shape, config):
    """
    Creates a Convolutional Autoencoder.
    """
    latent_dim = config.get('latent_dim', 128)
    
    # Encoder
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(x)
    shape_before_flattening = tf.keras.backend.int_shape(x)[1:]
    
    x = layers.Flatten()(x)
    latent = layers.Dense(latent_dim, name='latent_vector')(x)
    
    encoder = models.Model(inputs, latent, name='encoder')
    
    # Decoder
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(tf.math.reduce_prod(shape_before_flattening))(latent_inputs)
    x = layers.Reshape(shape_before_flattening)(x)
    
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=2)(x)
    
    outputs = layers.Conv2DTranspose(input_shape[-1], (3, 3), activation='sigmoid', padding='same')(x)
    
    decoder = models.Model(latent_inputs, outputs, name='decoder')
    
    # Autoencoder
    autoencoder_outputs = decoder(encoder(inputs))
    autoencoder = models.Model(inputs, autoencoder_outputs, name='autoencoder')
    
    return autoencoder
