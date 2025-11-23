import tensorflow as tf
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_tflite(model_path, output_path, quantization='float16'):
    """
    Converts a Keras model to TFLite.
    """
    logger.info(f"Converting {model_path} to TFLite...")
    
    try:
        model = tf.keras.models.load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        if quantization == 'float16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif quantization == 'int8':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # Requires representative dataset for full int8
            
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
            
        logger.info(f"Model saved to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    convert_to_tflite("checkpoints/cnn_custom_final.h5", "inference/model.tflite")
