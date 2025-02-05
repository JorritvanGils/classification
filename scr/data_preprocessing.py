from datasets import load_dataset
import os
import shutil
from tqdm import tqdm

# Define relative paths
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print('base_path:', base_path)
data_path = os.path.join(base_path, "data")
print('data_path:', data_path)
raw_data_path = os.path.join(data_path, "raw")
print('raw_data_path:', raw_data_path)
processed_data_path = os.path.join(data_path, "processed")
print('processed_data_path:', processed_data_path)

def download_imagenet(output_dir=raw_data_path):
    """Download the ImageNet-1K dataset to a custom directory."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading ImageNet-1K dataset to: {output_dir}")
    dataset = load_dataset("imagenet-1k", split="train", trust_remote_code=True, cache_dir=output_dir)
    
    return dataset

def filter_vehicles(dataset, vehicle_classes, output_dir=processed_data_path):
    """Filter the dataset to include only vehicle-related classes."""
    os.makedirs(output_dir, exist_ok=True)
    
    filtered_data = []
    for item in tqdm(dataset, desc="Filtering vehicles"):
        if item["label"] in vehicle_classes:
            filtered_data.append(item)
    
    return filtered_data

def save_filtered_dataset(filtered_data, output_dir=processed_data_path):
    """Save the filtered dataset as image files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for item in tqdm(filtered_data, desc="Saving images"):
        img_path = os.path.join(output_dir, f"{item['label']}_{item['image'].filename}")
        item['image'].save(img_path)

if __name__ == "__main__":
    dataset = download_imagenet()
    vehicle_classes = {"car", "bus", "truck", "motorcycle", "bicycle"}  # Replace with actual class indices
    # filtered_data = filter_vehicles(dataset, vehicle_classes)
    # save_filtered_dataset(filtered_data)
