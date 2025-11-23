from .cnn_custom import create_cnn_custom
from .transfer_learning import create_transfer_model
from .autoencoder import create_autoencoder

def get_model(model_name, input_shape, num_classes, config):
    """
    Factory function to get a model by name.
    """
    model_name = model_name.lower()
    
    if model_name == 'cnn_custom':
        return create_cnn_custom(input_shape, num_classes, config.get('cnn_custom', {}))
    
    elif model_name in ['resnet50', 'efficientnetb0', 'vgg16']:
        return create_transfer_model(model_name, input_shape, num_classes, config.get(model_name, {}))
        
    elif model_name == 'autoencoder':
        return create_autoencoder(input_shape, config.get('autoencoder', {}))
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")
