
import os

import unittest

import numpy as np
import pandas as pd

from glob import glob

from src.preprocess import (
    load_image,
    calibrate_image,
    preprocess_image,
    process_spectral_band,
    process_environmental_metadata,
    get_augmentation_pipeline
)

DATA_DIR = "../data/raw/"
PROCESSED_DIR = "../data/processed/"
IMAGE_SIZE = (256, 256)
SUPPORTED_SPECTRAL_BANDS = ['UV', 'VIS', 'NIR', 'SWIR', 'LWIR']

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        self.sample_image_path = os.path.join(DATA_DIR, "sample_image.png")
        self.sample_metadata_path = os.path.join(DATA_DIR, "environmental_metadata.csv")

        # Create a sample image
        sample_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        from cv2 import imwrite
        imwrite(self.sample_image_path, sample_image)

        # Create sample metadata
        sample_metadata = pd.DataFrame({
            "temperature": [25, 26, np.nan, 27],
            "humidity": [60, 65, 63, np.nan]
        })
        sample_metadata.to_csv(self.sample_metadata_path, index=False)

    def tearDown(self):
        # Clean up test files
        if os.path.exists(self.sample_image_path):
            os.remove(self.sample_image_path)
        if os.path.exists(self.sample_metadata_path):
            os.remove(self.sample_metadata_path)

    def test_load_image(self):
        image = load_image(self.sample_image_path)
        self.assertIsNotNone(image)
        self.assertEqual(image.shape, (256, 256, 3))

    def test_calibrate_image(self):
        image = load_image(self.sample_image_path)
        calibrated_image = calibrate_image(image)
        self.assertEqual(calibrated_image.shape, (256, 256))

    def test_preprocess_image(self):
        transform_pipeline = get_augmentation_pipeline(IMAGE_SIZE)
        preprocessed_image = preprocess_image(self.sample_image_path, transform_pipeline)
        self.assertEqual(preprocessed_image.shape, (3, 256, 256))

    def test_process_environmental_metadata(self):
        processed_metadata = process_environmental_metadata(self.sample_metadata_path)
        self.assertIsInstance(processed_metadata, pd.DataFrame)
        self.assertFalse(processed_metadata.isnull().values.any())

    def test_process_spectral_band(self):
        test_band = 'UV'
        band_dir = os.path.join(DATA_DIR, test_band)
        os.makedirs(band_dir, exist_ok=True)

        # Create a sample image in the band directory
        sample_band_image_path = os.path.join(band_dir, "band_sample_image.png")
        from cv2 import imwrite
        imwrite(sample_band_image_path, np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8))

        transform_pipeline = get_augmentation_pipeline(IMAGE_SIZE)
        process_spectral_band(band_dir, test_band, transform_pipeline)

        processed_image_path = os.path.join(PROCESSED_DIR, test_band, "band_sample_image.npy")
        self.assertTrue(os.path.exists(processed_image_path))

        # Clean up
        if os.path.exists(processed_image_path):
            os.remove(processed_image_path)
        if os.path.exists(sample_band_image_path):
            os.remove(sample_band_image_path)
        if os.path.exists(band_dir):
            os.rmdir(band_dir)

if __name__ == '__main__':
    unittest.main()