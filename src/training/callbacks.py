import tensorflow as tf
import os
import logging

logger = logging.getLogger(__name__)

class TrainingLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logger.info(f"Epoch {epoch+1}: {logs}")

def get_callbacks(model_name, config):
    """
    Returns a list of callbacks.
    """
    training_config = config.get('training', {})
    log_dir = os.path.join(training_config.get('tensorboard_log_dir', 'logs/fit'), model_name)
    checkpoint_dir = training_config.get('checkpoint_dir', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f"{model_name}_best.h5"),
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=training_config.get('early_stopping_patience', 10),
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=training_config.get('reduce_lr_factor', 0.2),
            patience=training_config.get('reduce_lr_patience', 5),
            min_lr=training_config.get('min_lr', 1e-6)
        ),
        TrainingLogger()
    ]
    return callbacks
