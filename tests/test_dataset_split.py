import unittest
import os
import shutil
import tempfile
import pandas as pd
from src.preprocessing.dataset import load_and_split_data

class TestDatasetSplit(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        
        # Create dummy structure
        # category1/train/good/img1.png
        # category1/test/good/img1.png
        # category1/test/defect/img1.png
        
        self.categories = ['cat1', 'cat2']
        for cat in self.categories:
            os.makedirs(os.path.join(self.test_dir, cat, 'train', 'good'))
            os.makedirs(os.path.join(self.test_dir, cat, 'test', 'good'))
            os.makedirs(os.path.join(self.test_dir, cat, 'test', 'defect'))
            
            # Create dummy files
            # 100 normal images total per category to make math easy
            # 80 in train/good, 20 in test/good
            for i in range(80):
                open(os.path.join(self.test_dir, cat, 'train', 'good', f'img_{i}.png'), 'w').close()
            for i in range(20):
                open(os.path.join(self.test_dir, cat, 'test', 'good', f'img_{i}.png'), 'w').close()
                
            # 20 anomaly images per category
            for i in range(20):
                open(os.path.join(self.test_dir, cat, 'test', 'defect', f'img_{i}.png'), 'w').close()
                
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_split_ratios(self):
        # Total per category: 100 normal + 20 anomaly = 120
        # Total global: 240
        # Split: 0.8, 0.1, 0.1
        
        train_df, val_df, test_df, class_names = load_and_split_data(self.test_dir, split_ratios=(0.8, 0.1, 0.1))
        
        total = len(train_df) + len(val_df) + len(test_df)
        self.assertEqual(total, 240)
        
        # Check sizes (approximate due to stratification rounding)
        # Train should be ~192 (0.8 * 240)
        # Val should be ~24
        # Test should be ~24
        
        self.assertTrue(190 <= len(train_df) <= 194, f"Train size {len(train_df)} not expected")
        self.assertTrue(22 <= len(val_df) <= 26, f"Val size {len(val_df)} not expected")
        self.assertTrue(22 <= len(test_df) <= 26, f"Test size {len(test_df)} not expected")
        
        # Check class names
        # We expect: cat1_good, cat1_defect, cat2_good, cat2_defect
        self.assertEqual(len(class_names), 4)
        self.assertIn('cat1_good', class_names)
        self.assertIn('cat1_defect', class_names)
        self.assertIn('cat2_good', class_names)
        self.assertIn('cat2_defect', class_names)
        
    def test_stratification(self):
        train_df, val_df, test_df, class_names = load_and_split_data(self.test_dir, split_ratios=(0.8, 0.1, 0.1))
        
        # Check if we have all classes in all sets (might fail for very small sets, but with 20 defects we expect at least 1 in val/test)
        # 20 defects -> 10% = 2. So we expect at least 1-2 defects in val/test.
        
        for df, name in [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
            unique_labels = df['label'].unique()
            # We expect 4 classes
            self.assertEqual(len(unique_labels), 4, f"Not all classes present in {name} set")
            
        # Verify label strings match labels
        sample = train_df.iloc[0]
        label_idx = sample['label']
        label_str = sample['label_str']
        self.assertEqual(class_names[label_idx], label_str)

    def test_target_category(self):
        # Test loading only 'cat1'
        train_df, val_df, test_df, class_names = load_and_split_data(self.test_dir, split_ratios=(0.8, 0.1, 0.1), target_category='cat1')
        
        # Should only have cat1 classes
        self.assertEqual(len(class_names), 2) # cat1_good, cat1_defect
        self.assertIn('cat1_good', class_names)
        self.assertIn('cat1_defect', class_names)
        self.assertNotIn('cat2_good', class_names)
        
        # Check total size (100 + 20 = 120)
        total = len(train_df) + len(val_df) + len(test_df)
        self.assertEqual(total, 120)
        
        # Test invalid category
        with self.assertRaises(ValueError):
            load_and_split_data(self.test_dir, target_category='non_existent')

if __name__ == '__main__':
    unittest.main()
