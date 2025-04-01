import cv2
import numpy as np
import os
import random

def load_and_save_dataset(simulated_dir, corrected_dir, output_dir="data/processed", val_ratio=0.2):
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image paths with types
    image_data = []
    type_map = {"protanopia": 0, "deuteranopia": 1, "tritanopia": 2}
    
    for cb_type, type_idx in type_map.items():
        sim_path = os.path.join(simulated_dir, cb_type)
        corr_path = os.path.join(corrected_dir, cb_type)
        
        for img_name in os.listdir(sim_path):
            sim_img_path = os.path.join(sim_path, img_name)
            corr_img_path = os.path.join(corr_path, img_name)
            if os.path.exists(corr_img_path):
                image_data.append((sim_img_path, corr_img_path, type_idx))
    
    # Shuffle and split
    random.shuffle(image_data)
    split_idx = int(len(image_data) * (1 - val_ratio))
    train_data = image_data[:split_idx]
    val_data = image_data[split_idx:]
    
    # Process batches
    def process_batch(data):
        X, y, types = [], [], []
        for sim_path, corr_path, type_idx in data:
            sim_img = cv2.imread(sim_path)
            corr_img = cv2.imread(corr_path)
            if sim_img is not None and corr_img is not None:
                X.append(cv2.resize(sim_img, (256, 256)))
                y.append(cv2.resize(corr_img, (256, 256)))
                types.append(type_idx)
        return np.array(X), np.array(y), np.array(types)
    
    X_train, y_train, types_train = process_batch(train_data)
    X_val, y_val, types_val = process_batch(val_data)
    
    # Save as numpy arrays
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "types_train.npy"), types_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    np.save(os.path.join(output_dir, "types_val.npy"), types_val)
    
    print(f"Dataset prepared! Train: {len(X_train)} samples, Val: {len(X_val)} samples")

if __name__ == "__main__":
    load_and_save_dataset("data/simulated", "data/corrected")