import os
import torch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from typing import Tuple
from scipy.ndimage import laplace
import unittest


class MNISTProcessor:
    """
    Class to handle downloading, rotating, filtering (blur & intensity), and exporting the MNIST dataset.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        rotation_degrees: Tuple[int, int] = (-90, 90),
        save_csv_path: str = "./data/rotated_mnist.csv",
        blur_threshold: float = 0.002,
        intensity_threshold: float = 0.1,
    ):
        self.data_dir = data_dir
        self.rotation_degrees = rotation_degrees
        self.save_csv_path = save_csv_path
        self.blur_threshold = blur_threshold
        self.intensity_threshold = intensity_threshold
        self.rotation_transform = transforms.RandomRotation(degrees=self.rotation_degrees)

    def load_dataset(self):
        """Download or load MNIST dataset."""
        print("📥 Loading MNIST dataset...")
        self.dataset = datasets.MNIST(
            self.data_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )

    def apply_rotation(self):
        """Apply random rotation to all images."""
        print("🔄 Applying rotation to images...")
        self.rotated_data = [
            (self.rotation_transform(img), label)
            for img, label in self.dataset
        ]

    def is_blurry(self, img: torch.Tensor) -> bool:
        """Detect blur using Laplacian variance."""
        laplacian_var = np.var(laplace(img.squeeze().numpy()))
        return laplacian_var < self.blur_threshold

    def filter_images(self):
        """
        Filter images based on blur and intensity.
        """
        print("🧹 Filtering images...")
        self.filtered_data = []
        for img, label in self.rotated_data:
            if img.mean().item() < self.intensity_threshold:
                continue
            if self.is_blurry(img):
                continue
            self.filtered_data.append((img, label))

        print(f"✅ Remaining samples after filtering: {len(self.filtered_data)}")

    def export_to_csv(self):
        """
        Save rotated and filtered data to CSV: each row is a flattened image + label.
        """
        print(f"💾 Saving processed data to: {self.save_csv_path}")
        image_vectors = [
            img.view(-1).numpy()
            for img, _ in self.filtered_data
        ]
        labels = [
            label
            for _, label in self.filtered_data
        ]
        df = pd.DataFrame(image_vectors)
        df["label"] = labels
        os.makedirs(os.path.dirname(self.save_csv_path), exist_ok=True)
        df.to_csv(self.save_csv_path, index=False)
        print("✅ Data saved successfully.")

    def run_full_pipeline(self):
        """Convenience method to run all steps."""
        self.load_dataset()
        self.apply_rotation()
        self.filter_images()
        self.export_to_csv()


# Optional tests using unittest
class TestMNISTProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = MNISTProcessor()
        self.processor.load_dataset()

    def test_dataset_load(self):
        self.assertTrue(len(self.processor.dataset) > 0)

    def test_rotation_preserves_shape(self):
        self.processor.apply_rotation()
        for img, _ in self.processor.rotated_data[:10]:
            self.assertEqual(img.shape, (1, 28, 28))

    def test_blur_detection(self):
        sample_img, _ = self.processor.dataset[0]
        blurry = self.processor.is_blurry(sample_img)
        self.assertIsInstance(blurry, bool)


if __name__ == "__main__":
    processor = MNISTProcessor()
    processor.run_full_pipeline()
    # To run tests: uncomment the line below
    unittest.main()
