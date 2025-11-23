import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def create_cnn_custom(input_shape, num_classes, config):
    """
    Creates a custom CNN model for anomaly classification.
    """
    filters = config.get('conv_filters', [32, 64, 128, 256])
    dropout_rate = config.get('dropout_rate', 0.5)
    
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    # Convolutional blocks
    for f in filters:
        x = layers.Conv2D(f, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)
        
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="cnn_custom")
    return model
