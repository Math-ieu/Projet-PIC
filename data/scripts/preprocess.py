import os
import glob
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config
with open("configs/model_configs.yaml", "r") as f:
    model_configs = yaml.safe_load(f)
    
IMG_SIZE = tuple(model_configs['cnn_custom']['input_shape'][:2])
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() 
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(image, label, category, mask=None):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    feature = {
        'image': _bytes_feature(image),
        'label': _int64_feature(label), # 0 for good, 1 for anomaly
        'category': _bytes_feature(category.encode('utf-8')),
    }
    if mask is not None:
        feature['mask'] = _bytes_feature(mask)
        
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def preprocess_image(image_path, grayscale=False):
    img = Image.open(image_path)
    if grayscale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    img = img.resize(IMG_SIZE)
    img_bytes = img.tobytes() # Raw bytes
    return img_bytes

def create_tfrecords(dataset_type='train'):
    """
    Create TFRecords for train or test sets.
    MVTec structure: category/train/good/xxx.png, category/test/defect_type/xxx.png
    """
    categories = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]
    
    output_dir = os.path.join(PROCESSED_DATA_DIR, dataset_type)
    os.makedirs(output_dir, exist_ok=True)
    
    for category in categories:
        logger.info(f"Processing {category} - {dataset_type}...")
        
        # Determine if grayscale (texture categories are often handled differently, but MVTec is mixed)
        # For simplicity, we convert everything to RGB as per config input_shape [256, 256, 3]
        # unless specified otherwise.
        
        record_file = os.path.join(output_dir, f"{category}.tfrecord")
        with tf.io.TFRecordWriter(record_file) as writer:
            
            category_path = os.path.join(RAW_DATA_DIR, category, dataset_type)
            if not os.path.exists(category_path):
                continue
                
            # Traverse subdirectories (e.g., 'good', 'crack', etc.)
            for defect_type in os.listdir(category_path):
                defect_path = os.path.join(category_path, defect_type)
                if not os.path.isdir(defect_path):
                    continue
                
                label = 0 if defect_type == 'good' else 1
                
                image_files = glob.glob(os.path.join(defect_path, "*.png"))
                for img_path in image_files:
                    img_bytes = preprocess_image(img_path)
                    
                    # Handle masks for test set anomalies
                    mask_bytes = None
                    if dataset_type == 'test' and label == 1:
                        mask_path = img_path.replace("/test/", "/ground_truth/").replace(".png", "_mask.png")
                        if os.path.exists(mask_path):
                            mask_bytes = preprocess_image(mask_path, grayscale=True)
                    
                    example = serialize_example(img_bytes, label, category, mask_bytes)
                    writer.write(example)

def main():
    if not os.path.exists(RAW_DATA_DIR):
        logger.error(f"{RAW_DATA_DIR} not found. Please run download_dataset.py first.")
        return

    create_tfrecords('train')
    create_tfrecords('test')
    logger.info("Preprocessing complete.")

if __name__ == "__main__":
    main()
