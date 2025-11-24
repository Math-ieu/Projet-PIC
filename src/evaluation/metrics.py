import tensorflow as tf
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import numpy as np
from skimage import measure

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

def calculate_iou(y_true, y_pred, threshold=0.5):
    """
    Calculate Intersection over Union (IoU) for segmentation masks.
    y_true: Ground truth masks (0 or 1)
    y_pred: Predicted anomaly maps (0 to 1)
    """
    y_pred_bin = (y_pred > threshold).astype(int)
    y_true = y_true.astype(int)
    
    intersection = np.logical_and(y_true, y_pred_bin).sum()
    union = np.logical_or(y_true, y_pred_bin).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
        
    return intersection / union

def calculate_pro(y_true, y_pred, threshold=0.5):
    """
    Calculate Per-Region Overlap (PRO).
    Average coverage of each connected component in ground truth.
    """
    y_pred_bin = (y_pred > threshold).astype(int)
    y_true = y_true.astype(int)
    
    # Label connected components in ground truth
    labeled_gt, num_features = measure.label(y_true, return_num=True, connectivity=2)
    
    if num_features == 0:
        # No anomalies in ground truth
        # If prediction is also empty, perfect. If prediction has noise, it's a false positive.
        # PRO is typically defined on anomalous regions. 
        # We return 1.0 if no anomalies exist (perfect coverage of "nothing").
        return 1.0
        
    pro_scores = []
    for region_idx in range(1, num_features + 1):
        region_mask = (labeled_gt == region_idx)
        region_area = region_mask.sum()
        
        # Intersection of prediction with this region
        intersection = np.logical_and(region_mask, y_pred_bin).sum()
        
        coverage = intersection / region_area
        pro_scores.append(coverage)
        
    return np.mean(pro_scores)
