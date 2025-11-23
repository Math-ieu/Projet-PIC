import tensorflow as tf
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import numpy as np

def get_metrics():
    """
    Returns a list of Keras metrics.
    """
    return [
        tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
    ]

def calculate_auc(y_true, y_pred):
    """
    Calculate AUC-ROC score.
    """
    # For multi-class, we might need one-vs-rest
    if len(np.unique(y_true)) > 2:
        return roc_auc_score(y_true, y_pred, multi_class='ovr')
    return roc_auc_score(y_true, y_pred)

def calculate_f1(y_true, y_pred_classes):
    """
    Calculate F1 score.
    """
    return f1_score(y_true, y_pred_classes, average='macro')

def get_confusion_matrix(y_true, y_pred_classes):
    """
    Calculate confusion matrix.
    """
    return confusion_matrix(y_true, y_pred_classes)
