import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import v2

def compute_mean_std(input_dir):
    """
    Compute and save/load the mean and standard deviation of input images.

    Args:
        input_dir (str): The directory containing the input images.

    Returns:
        tuple: The mean and standard deviation of the input images.
    """
    # Generate dynamic save path
    dir_name = os.path.basename(os.path.normpath(input_dir))
    save_path = f"mean_std_{dir_name}.json"

    # If the file exists, load existing values
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            data = json.load(f)
        mean, std = np.array(data["mean"]), np.array(data["std"])
        print(f"Loaded saved mean and std from {save_path}: {mean}, {std}")
        return mean, std

    print(f"Computing mean and std for images in {input_dir}...")

    # Initialize variables
    pixel_sum = np.zeros(3)
    pixel_squared_sum = np.zeros(3)
    num_pixels = 0

    # Collect all image paths for progress tracking
    image_paths = []
    for subfolder in os.listdir(input_dir):
        subfolder_path = os.path.join(input_dir, subfolder)
        if os.path.isdir(subfolder_path):
            for image_name in os.listdir(subfolder_path):
                image_paths.append(os.path.join(subfolder_path, image_name))

    # Process images
    for image_path in tqdm(image_paths, desc="Processing Images", unit="image"):
        try:
            image = Image.open(image_path).convert("RGB")
            image_array = np.array(image) / 255.0  # Normalize to [0,1]
            pixel_sum += np.sum(image_array, axis=(0, 1))
            pixel_squared_sum += np.sum(np.square(image_array), axis=(0, 1))
            num_pixels += image_array.shape[0] * image_array.shape[1]
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    if num_pixels == 0:
        raise ValueError("No valid images found for computing mean and std.")

    # Compute mean and std
    mean = pixel_sum / num_pixels
    std = np.sqrt((pixel_squared_sum / num_pixels) - np.square(mean))

    # Save computed values
    with open(save_path, "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)

    print(f"Computed and saved mean and std to {save_path}: {mean}, {std}")
    return mean, std


def create_transformers(
    mean, std, height, width, random_rotation_degrees, random_affine_degrees,
    random_translation, brightness, contrast, saturation, hue
):
    """
    Creates and returns training and testing data transformation pipelines.

    Args:
        mean (list or tuple): Mean values for normalization.
        std (list or tuple): Standard deviation values for normalization.
        height (int): Height to resize the images to.
        width (int): Width to resize the images to.
        random_rotation_degrees (int or float): Degrees for random rotation.
        random_affine_degrees (int or float): Degrees for random affine transformation.
        random_translation (tuple): Max absolute fraction for horizontal and vertical translations.
        brightness (float or tuple): Brightness factor for color jitter.
        contrast (float or tuple): Contrast factor for color jitter.
        saturation (float or tuple): Saturation factor for color jitter.
        hue (float or tuple): Hue factor for color jitter.

    Returns:
        tuple: A tuple containing the training and testing transformation pipelines.
    """
    train_transform = v2.Compose([
        v2.Resize((height, width)),
        v2.CenterCrop(height),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(random_rotation_degrees),
        v2.RandomAffine(
            random_affine_degrees,
            translate=random_translation
        ),
        v2.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        ),
        v2.ToTensor(),
        v2.Normalize(mean=mean, std=std)
    ])

    test_transform = v2.Compose([
        v2.Resize((height, width)),
        v2.CenterCrop(height),
        v2.ToTensor(),
        v2.Normalize(mean=mean, std=std)
    ])

    return train_transform, test_transform
