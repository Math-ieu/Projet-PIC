import os
import glob
import tensorflow as tf
from src.training.config import load_training_config, load_model_config
from src.models.model_factory import get_model
from src.training.trainer import Trainer
from src.evaluation.benchmark import evaluate_model
from src.preprocessing.dataset import load_and_split_data
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dataset_from_dataframe(df, batch_size, image_size):
    """
    Convert a pandas DataFrame with 'filepath' and 'label' columns to a tf.data.Dataset.
    """
    if df.empty:
        return None
        
    filepaths = df['filepath'].values
    labels = df['label'].values
    
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    
    def load_image(filepath, label):
        img = tf.io.read_file(filepath)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, image_size)
        # Normalize to [0, 1]
        img = img / 255.0
        return img, label
        
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def main():
    training_config = load_training_config()
    model_config = load_model_config()
    
    data_cfg = training_config['data']
    train_cfg = training_config['training']
    
    # Get all categories
    categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    categories.sort()
    
    if not categories:
        logger.error("No categories found! Please run download_dataset.py first.")
        return

    logger.info(f"Found {len(categories)} categories: {categories}")

    results = {}
    
    # Loop over each category
    for category in categories:
        logger.info(f"=== Processing Category: {category} ===")
        
        logger.info(f"Loading and splitting data for {category}...")
        # 80% train, 10% val, 10% test
        try:
            train_df, val_df, test_df, class_names = load_and_split_data(
                data_dir, 
                split_ratios=(0.8, 0.1, 0.1),
                target_category=category,
                augment=True
            )
        except Exception as e:
            logger.error(f"Error loading data for {category}: {e}")
            continue
        
        if train_df.empty:
            logger.warning(f"No data found for {category}, skipping.")
            continue

        num_classes = len(class_names)
        logger.info(f"Number of classes for {category}: {num_classes}")
        logger.info(f"Classes: {class_names}")

        # Create TF Datasets
        batch_size = data_cfg['batch_size']
        image_size = tuple(data_cfg['image_size'])
        
        train_ds = create_dataset_from_dataframe(train_df, batch_size, image_size)
        val_ds = create_dataset_from_dataframe(val_df, batch_size, image_size)
        test_ds = create_dataset_from_dataframe(test_df, batch_size, image_size)
        
        # Models to train
        models_to_train = ['cnn_custom', 'resnet50', 'autoencoder']
        
        for model_name in models_to_train:
            run_name = f"{model_name}_{category}"
            logger.info(f"Starting training for {run_name}...")
            
            try:
                # Create Model
                # Multi-class classification
                
                model = get_model(
                    model_name, 
                    input_shape=image_size + (3,), 
                    num_classes=num_classes, 
                    config=model_config
                )
                
                trainer = Trainer(model, model_name, training_config, run_name=run_name)
                trainer.compile(learning_rate=model_config.get(model_name, {}).get('learning_rate', 0.001))
                
                logger.info(f"Training {run_name}...")
                history = trainer.train(train_ds, val_ds, epochs=model_config.get(model_name, {}).get('epochs', 10))
                
                # Evaluate on test set
                logger.info(f"Evaluating {run_name} on test set...")
                test_loss, test_acc = model.evaluate(test_ds)
                
                results[run_name] = {
                    'status': 'trained', 
                    'category': category,
                    'test_accuracy': test_acc,
                    'test_loss': test_loss
                }
                
                # Save model
                os.makedirs("checkpoints", exist_ok=True)
                model.save(f"checkpoints/{run_name}_final.h5")
                
            except Exception as e:
                logger.error(f"Failed to train {run_name}: {e}")
                results[run_name] = {'status': 'failed', 'error': str(e)}
            
    # Save results
    with open("training_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    logger.info("All training jobs completed.")

if __name__ == "__main__":
    main()

