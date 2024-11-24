# algorithm_ds.py

import numpy as np
import os
from PIL import Image

class ImageProcessor:
    def __init__(self, img_size=(64, 64), threshold=0.3):
        self.img_size = img_size
        self.threshold = threshold

    def process_image(self, img_path):
        """
        Load and preprocess an image.
        """
        try:
            img = Image.open(img_path).convert("L")
            img = img.resize(self.img_size)
            img_array = np.array(img) / 255.0
            binary_img = (img_array > self.threshold).astype(np.float32)
            return binary_img
        except Exception as e:
            raise ValueError(f"Error processing image {img_path}: {e}")

class DataManager:
    def __init__(self, base_dir, img_processor):
        self.base_dir = base_dir
        self.img_processor = img_processor
        self.data_by_class = {}

    def load_data_from_folders(self):
        """
        Load and preprocess images from folders.
        """
        label_names = ['0', '1', '3', '5', '6', '9']
        data_by_class = {label: [] for label in label_names}

        for label_name in label_names:
            label_path = os.path.join(self.base_dir, label_name)
            if not os.path.isdir(label_path):
                continue
            for filename in os.listdir(label_path):
                if filename.lower().endswith(('.jpg', '.png')):
                    img_path = os.path.join(label_path, filename)
                    try:
                        binary_img = self.img_processor.process_image(img_path)
                        data_by_class[label_name].append(binary_img)
                    except ValueError as e:
                        print(e)

        self.data_by_class = data_by_class
        return self.data_by_class

class ProbabilityTrainer:
    def __init__(self, data_by_class):
        self.data_by_class = data_by_class
        self.class_pixel_probs = {}
        self.class_counts = {}
        self.total_images = 0
        self.class_priors = {}

    def train_pixel_probabilities(self):
        """
        Train pixel probabilities for each class.
        """
        for label, images in self.data_by_class.items():
            if len(images) == 0:
                continue
            image_stack = np.stack(images)
            pixel_prob = (np.sum(image_stack, axis=0) + 1) / (len(images) + 2)
            self.class_pixel_probs[label] = pixel_prob
            self.class_counts[label] = len(images)
            self.total_images += len(images)

    def calculate_class_priors(self):
        """
        Calculate class priors.
        """
        self.class_priors = {label: count / self.total_images for label, count in self.class_counts.items()}

    def get_class_pixel_probs(self):
        return self.class_pixel_probs

    def get_class_priors(self):
        return self.class_priors

class MapHandler:
    def __init__(self):
        pass

    @staticmethod
    def load_map(file_path):
        """
        Load map data from a file.
        """
        return np.load(file_path)

    @staticmethod
    def display_map_with_rover(map_array, rover_position, visited_white, visited_black, ax):
        """
        Visualize the map with rover and visited cells.
        """
        display_map = np.full_like(map_array, 0.5, dtype=np.float32)
        for r, c in visited_white:
            display_map[r, c] = 1.0
        for r, c in visited_black:
            display_map[r, c] = 0.0
        ax.clear()
        ax.imshow(display_map, cmap="gray", vmin=0, vmax=1)
        ax.scatter(rover_position[1], rover_position[0], c="red", label="Rover")
        ax.set_title("Rover Exploration")
        ax.axis("off")

class PosteriorCalculator:
    def __init__(self, class_pixel_probs, class_priors):
        self.class_pixel_probs = class_pixel_probs
        self.class_priors = class_priors

    def calculate_posterior_probs(self, visited_white, map_array, sorted_labels, smoothing=1e-2):
        """
        Calculate posterior probabilities for each class.
        """
        observed = np.zeros_like(map_array, dtype=np.float32)
        for r, c in visited_white:
            observed[r, c] = 1

        num_classes = len(sorted_labels)
        posterior_probs = np.zeros(num_classes)
        for idx, label in enumerate(sorted_labels):
            likelihood = (
                observed * np.log(self.class_pixel_probs[label] + smoothing) +
                (1 - observed) * np.log(1 - self.class_pixel_probs[label] + smoothing)
            )
            posterior_probs[idx] = likelihood.sum()
        posterior_probs = np.exp(posterior_probs - np.max(posterior_probs))
        posterior_probs *= [self.class_priors[label] for label in sorted_labels]
        posterior_probs /= posterior_probs.sum()
        return posterior_probs 
