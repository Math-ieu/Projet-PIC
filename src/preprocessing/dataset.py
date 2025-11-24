import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import tensorflow as tf
import shutil
import random
import numpy as np

logger = logging.getLogger(__name__)

def augment_image(image_path, save_dir, prefix, count):
    """
    Reads an image, applies random augmentations, and saves it.
    """
    try:
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=3)
        
        # Random augmentations
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        
        # Ensure valid range [0, 255]
        img = tf.clip_by_value(img, 0, 255)
        img = tf.cast(img, tf.uint8)
        
        encoded_img = tf.image.encode_png(img)
        
        filename = f"{prefix}_{count}.png"
        save_path = os.path.join(save_dir, filename)
        
        tf.io.write_file(save_path, encoded_img)
        return save_path
    except Exception as e:
        logger.warning(f"Failed to augment image {image_path}: {e}")
        return None

def balance_classes(df, target_count=1000, save_root="data/processed/augmented"):
    """
    Balances classes in the DataFrame by augmenting rare classes to reach target_count.
    """
    if df.empty:
        return df
        
    logger.info(f"Balancing classes to target count: {target_count}...")
    
    # Create augmentation directory
    if os.path.exists(save_root):
        # Optional: Clear previous augmentations to avoid staleness, 
        # but might be slow if we re-run often. For now, let's keep it simple and overwrite/add.
        pass
    else:
        os.makedirs(save_root, exist_ok=True)
        
    new_rows = []
    
    # Group by label (which maps to a specific Category_DefectType)
    # We need to know the label_str to name files appropriately
    unique_labels = df['label'].unique()
    
    for label in unique_labels:
        class_subset = df[df['label'] == label]
        current_count = len(class_subset)
        
        if current_count >= target_count:
            continue
            
        needed = target_count - current_count
        label_str = class_subset.iloc[0]['label_str']
        category = class_subset.iloc[0]['category']
        
        logger.info(f"Augmenting class '{label_str}': {current_count} -> {target_count} (+{needed})")
        
        # Create class specific save dir
        class_save_dir = os.path.join(save_root, label_str)
        os.makedirs(class_save_dir, exist_ok=True)
        
        # Source images to augment
        source_images = class_subset['filepath'].tolist()
        
        for i in range(needed):
            # Randomly select a source image
            src_img = random.choice(source_images)
            
            # Augment and save
            new_path = augment_image(src_img, class_save_dir, "aug", i)
            
            if new_path:
                new_rows.append({
                    'filepath': new_path,
                    'category': category,
                    'label': label,
                    'label_str': label_str
                })
                
    if new_rows:
        augmented_df = pd.DataFrame(new_rows)
        df = pd.concat([df, augmented_df], ignore_index=True)
        
    logger.info(f"Balancing complete. Total samples: {len(df)}")
    return df

def load_and_split_data(data_dir, split_ratios=(0.8, 0.1, 0.1), seed=42, target_category=None, augment=False):
    """
    Load MVTec AD data, merge normal and abnormal, and split into train/val/test.
    Assigns unique labels for each (Category, DefectType) pair.
    
    Args:
        data_dir (str): Path to the root of the MVTec AD dataset (containing category folders).
        split_ratios (tuple): (train_ratio, val_ratio, test_ratio). Must sum to 1.
        seed (int): Random seed for reproducibility.
        seed (int): Random seed for reproducibility.
        target_category (str, optional): If provided, only load data for this specific category.
        augment (bool): If True, augment the training set to balance classes (1000 samples/class).
        
    Returns:
        tuple: (train_df, val_df, test_df, class_names)
               Each df has columns ['filepath', 'category', 'label', 'label_str']
               class_names: list of string labels indexed by the label integer.
    """
    if sum(split_ratios) != 1.0:
        raise ValueError("Split ratios must sum to 1.0")
        
    train_ratio, val_ratio, test_ratio = split_ratios
    
    # 1. Collect all data and identify classes
    data = []
    
    # Get all categories (subdirectories in data_dir)
    if target_category:
        if not os.path.isdir(os.path.join(data_dir, target_category)):
            raise ValueError(f"Category '{target_category}' not found in {data_dir}")
        categories = [target_category]
    else:
        categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        categories.sort() # Ensure deterministic order
    
    # First pass: Collect all unique label strings to build mapping
    # We need to scan to find all defect types
    unique_labels = set()
    
    for category in categories:
        cat_dir = os.path.join(data_dir, category)
        
        # Train data (only 'good')
        unique_labels.add(f"{category}_good")
        
        # Test data (contains 'good' and various defect types)
        test_dir = os.path.join(cat_dir, 'test')
        if os.path.exists(test_dir):
            for defect_type in os.listdir(test_dir):
                if os.path.isdir(os.path.join(test_dir, defect_type)):
                    unique_labels.add(f"{category}_{defect_type}")
                    
    # Create class mapping
    class_names = sorted(list(unique_labels))
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    logger.info(f"Found {len(class_names)} unique classes: {class_names}")
    
    # Second pass: Collect data with labels
    for category in categories:
        cat_dir = os.path.join(data_dir, category)
        
        # Train data (only 'good' usually in MVTec AD train set)
        train_good_dir = os.path.join(cat_dir, 'train', 'good')
        if os.path.exists(train_good_dir):
            label_str = f"{category}_good"
            label = class_to_idx[label_str]
            for img_path in glob.glob(os.path.join(train_good_dir, '*.png')):
                data.append({
                    'filepath': img_path,
                    'category': category,
                    'label': label,
                    'label_str': label_str
                })
                
        # Test data (contains 'good' and various defect types)
        test_dir = os.path.join(cat_dir, 'test')
        if os.path.exists(test_dir):
            for defect_type in os.listdir(test_dir):
                defect_dir = os.path.join(test_dir, defect_type)
                if not os.path.isdir(defect_dir):
                    continue
                    
                label_str = f"{category}_{defect_type}"
                label = class_to_idx[label_str]
                
                for img_path in glob.glob(os.path.join(defect_dir, '*.png')):
                    data.append({
                        'filepath': img_path,
                        'category': category,
                        'label': label,
                        'label_str': label_str
                    })

    df = pd.DataFrame(data)
    
    if df.empty:
        logger.warning(f"No data found in {data_dir}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    # 2. Stratified Split
    # We want to stratify by Label (which now encodes both Category and DefectType)
    # However, some defect types might have very few samples (e.g. < 3), which makes stratification impossible.
    # We should fall back to simple random split or warn if stratification fails.
    
    # Check class counts
    class_counts = df['label'].value_counts()
    rare_classes = class_counts[class_counts < 2].index.tolist()
    
    if rare_classes:
        logger.warning(f"Classes {rare_classes} have fewer than 2 samples. Stratification for these will fail/be imperfect.")
        # For now, we proceed. train_test_split might error if a class has only 1 sample.
        # We can filter out single-sample classes or just duplicate them? 
        # Let's assume MVTec AD has enough samples per defect type (usually > 10).
    
    # First split: Train vs (Val + Test)
    test_val_ratio = val_ratio + test_ratio
    
    if test_val_ratio == 0:
        return df, pd.DataFrame(), pd.DataFrame(), class_names
        
    try:
        train_df, temp_df = train_test_split(
            df, 
            train_size=train_ratio, 
            stratify=df['label'], 
            random_state=seed
        )
    except ValueError as e:
        logger.warning(f"Stratified split failed (likely due to rare classes): {e}. Falling back to random split.")
        train_df, temp_df = train_test_split(
            df, 
            train_size=train_ratio, 
            random_state=seed
        )

    # 3. Augmentation (Balance Classes) - ONLY on Train set
    if augment:
        train_df = balance_classes(train_df, target_count=1000)
    
    # Second split: Val vs Test
    if test_ratio == 0:
        val_df = temp_df
        test_df = pd.DataFrame()
    elif val_ratio == 0:
        val_df = pd.DataFrame()
        test_df = temp_df
    else:
        relative_test_size = test_ratio / (val_ratio + test_ratio)
        try:
            val_df, test_df = train_test_split(
                temp_df,
                test_size=relative_test_size,
                stratify=temp_df['label'],
                random_state=seed
            )
        except ValueError:
             val_df, test_df = train_test_split(
                temp_df,
                test_size=relative_test_size,
                random_state=seed
            )
        
    logger.info(f"Data split complete.")
    logger.info(f"Train: {len(train_df)} images")
    logger.info(f"Val: {len(val_df)} images")
    logger.info(f"Test: {len(test_df)} images")
    
    return train_df, val_df, test_df, class_names

