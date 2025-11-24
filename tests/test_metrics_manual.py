import sys
import os
import numpy as np
import tensorflow as tf

sys.path.append(os.getcwd())

from src.evaluation.metrics import calculate_iou, calculate_pro
from src.evaluation.benchmark import evaluate_model

def test_metrics_functions():
    print("Testing metrics functions...")
    
    # Dummy masks (10x10)
    y_true = np.zeros((10, 10))
    y_true[2:5, 2:5] = 1 # A 3x3 square
    
    y_pred = np.zeros((10, 10))
    y_pred[2:5, 2:5] = 0.8 # Perfect overlap
    
    iou = calculate_iou(y_true, y_pred)
    print(f"IoU (Perfect): {iou}")
    assert iou == 1.0
    
    pro = calculate_pro(y_true, y_pred)
    print(f"PRO (Perfect): {pro}")
    assert pro == 1.0
    
    # Partial overlap
    y_pred_partial = np.zeros((10, 10))
    y_pred_partial[2:4, 2:5] = 0.8 # Missing one row
    
    iou_partial = calculate_iou(y_true, y_pred_partial)
    print(f"IoU (Partial): {iou_partial}")
    assert iou_partial < 1.0
    
    pro_partial = calculate_pro(y_true, y_pred_partial)
    print(f"PRO (Partial): {pro_partial}")
    # PRO should be 2/3 = 0.666... because 2/3 of the region is covered
    assert 0.6 < pro_partial < 0.7
    
    print("Metrics functions passed.")

def test_evaluate_model():
    print("Testing evaluate_model...")
    
    # Dummy model
    inputs = tf.keras.Input(shape=(10, 10, 1))
    outputs = tf.keras.layers.Lambda(lambda x: x)(inputs) # Identity model
    model = tf.keras.Model(inputs, outputs)
    model.compile(loss='mse')
    
    # Dummy dataset
    x = np.random.rand(5, 10, 10, 1).astype(np.float32)
    y = np.random.randint(0, 2, (5, 10, 10, 1)).astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(1)
    
    # Test with masks (y is treated as mask here)
    results = evaluate_model(model, ds, test_masks=y)
    print("Results:", results)
    
    assert 'iou' in results
    assert 'pro' in results
    
    print("evaluate_model passed.")

if __name__ == "__main__":
    test_metrics_functions()
    test_evaluate_model()
