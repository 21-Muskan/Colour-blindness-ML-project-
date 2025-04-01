import cv2
import numpy as np
import os

def rgb_to_lms(rgb_img):
    """Convert RGB to LMS using CIECAM02 matrix from Daltonize"""
    return np.dot(rgb_img, np.array([
        [0.7328, 0.4296, -0.1624],
        [-0.7036, 1.6975, 0.0061],
        [0.0030, 0.0136, 0.9834]
    ]).T)

def lms_to_rgb(lms_img):
    """Convert LMS to RGB using inverse CIECAM02 matrix"""
    return np.dot(lms_img, np.array([
        [1.096124, -0.278869, 0.182745],
        [0.454369, 0.473533, 0.072098],
        [-0.009628, -0.005698, 1.015326]
    ]).T)

def simulate_color_blindness(img_rgb, blindness_type="protanopia"):
    """Simulate color blindness using improved Daltonize matrices"""
    img_rgb = img_rgb.astype(np.float32) / 255.0
    lms = rgb_to_lms(img_rgb)
    
    # Daltonize's transformation matrices
    if blindness_type == "protanopia":
        transform = np.array([[0, 2.02344, -2.52581], [0, 1, 0], [0, 0, 1]])
    elif blindness_type == "deuteranopia":
        transform = np.array([[1, 0, 0], [0.494207, 0, 1.24827], [0, 0, 1]])
    elif blindness_type == "tritanopia":
        transform = np.array([[1, 0, 0], [0, 1, 0], [-0.395913, 0.801109, 0]])
    
    simulated_lms = np.dot(lms, transform.T)
    simulated_rgb = lms_to_rgb(simulated_lms)
    
    # Maintain luminance
    lab_orig = cv2.cvtColor((img_rgb*255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    lab_sim = cv2.cvtColor((simulated_rgb*255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    lab_sim[:,:,0] = lab_orig[:,:,0]  # Keep original luminance
    
    result = cv2.cvtColor(lab_sim, cv2.COLOR_LAB2RGB)
    return np.clip(result, 0, 255).astype(np.uint8)

def process_folder(input_dir, output_dir, blindness_type):
    """Process all images in a folder, skipping existing files"""
    os.makedirs(output_dir, exist_ok=True)
    processed = 0
    skipped = 0
    
    for img_name in os.listdir(input_dir):
        output_path = os.path.join(output_dir, img_name)
        
        if os.path.exists(output_path):
            skipped += 1
            continue
            
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        simulated = simulate_color_blindness(img_rgb, blindness_type)
        cv2.imwrite(output_path, cv2.cvtColor(simulated, cv2.COLOR_RGB2BGR))
        processed += 1
        print(f"Processed: {img_path} -> {output_path}")
    
    print(f"\n{blindness_type.upper()} Summary:")
    print(f"Processed: {processed} | Skipped: {skipped}")

def generate_all_simulations():
    """Generate simulations for all color blindness types"""
    input_dir = "data/original/"
    output_dirs = {
        "protanopia": "data/simulated/protanopia/",
        "deuteranopia": "data/simulated/deuteranopia/",
        "tritanopia": "data/simulated/tritanopia/"
    }
    
    print("Starting color blindness simulation...")
    for cb_type, out_dir in output_dirs.items():
        print(f"\nProcessing {cb_type.upper()}...")
        process_folder(input_dir, out_dir, cb_type)
    print("\nAll simulations completed!")

if __name__ == "__main__":
    generate_all_simulations()