import os
import json
import numpy as np
import tensorflow as tf
from .metrics import calculate_auc, calculate_f1
import logging

logger = logging.getLogger(__name__)

def evaluate_model(model, test_dataset):
    """
    Evaluates a model on the test dataset.
    """
    logger.info("Evaluating model...")
    results = model.evaluate(test_dataset, return_dict=True)
    
    # Get predictions for advanced metrics
    y_true = []
    y_pred = []
    
    for images, labels in test_dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(preds)
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate additional metrics
    try:
        auc = calculate_auc(y_true, y_pred) # Might fail if only one class in batch
        f1 = calculate_f1(y_true, y_pred_classes)
        results['auc'] = auc
        results['f1'] = f1
    except Exception as e:
        logger.warning(f"Could not calculate advanced metrics: {e}")
        
    return results

def compare_models(models_dict, test_dataset):
    """
    Compares multiple models.
    """
    comparison = {}
    for name, model in models_dict.items():
        logger.info(f"Benchmarking {name}...")
        comparison[name] = evaluate_model(model, test_dataset)
        
    return comparison
