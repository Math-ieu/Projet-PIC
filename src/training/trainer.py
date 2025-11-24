import tensorflow as tf
import os
from .callbacks import get_callbacks
from ..evaluation.metrics import get_metrics

class Trainer:
    def __init__(self, model, model_name, config, run_name=None):
        self.model = model
        self.model_name = model_name
        self.run_name = run_name
        self.config = config
        self.training_config = config.get('training', {})
        
    def compile(self, learning_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        if self.model_name == 'autoencoder':
            loss = 'mse' # Simplified for now, can add SSIM
            metrics = ['mae']
        else:
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
            
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
    def train(self, train_dataset, val_dataset, epochs):
        callbacks = get_callbacks(self.model_name, self.config, self.run_name)
        
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks
        )
        return history
