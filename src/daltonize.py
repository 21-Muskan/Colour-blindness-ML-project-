import cv2
import numpy as np
import os
from simulate import simulate_color_blindness

def daltonize(img_rgb, blindness_type="protanopia"):
    """Enhanced color correction using Daltonize's error amplification"""
    original = img_rgb.astype(np.float32) / 255.0
    simulated = simulate_color_blindness(img_rgb, blindness_type) / 255.0
    
    # Calculate error matrix
    err = original - simulated
    
    # Apply type-specific correction
    correction = np.zeros_like(original)
    if blindness_type == "protanopia":
        correction[:,:,0] = err[:,:,0] * 0.7  # Red
        correction[:,:,1] = err[:,:,1] * 1.1  # Green
    elif blindness_type == "deuteranopia":
        correction[:,:,0] = err[:,:,0] * 0.8  # Red
        correction[:,:,1] = err[:,:,1] * 1.2  # Green
    else:  # tritanopia
        correction[:,:,2] = err[:,:,2] * 1.0  # Blue
    
    # Combine with original and apply gamma
    corrected = np.clip(original + correction * 1.5, 0, 1)
    corrected = corrected ** 0.9  # Gamma adjustment
    
    return (corrected * 255).astype(np.uint8)

def process_folder(input_dir, output_dir, blindness_type):
    """Process folder with enhanced color correction"""
    os.makedirs(output_dir, exist_ok=True)
    processed = 0
    skipped = 0
    
    for img_name in os.listdir(input_dir):
        out_path = os.path.join(output_dir, img_name)
        
        if os.path.exists(out_path):
            skipped += 1
            continue
            
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        corrected = daltonize(img_rgb, blindness_type)
        cv2.imwrite(out_path, cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR))
        processed += 1
        print(f"Corrected: {img_path} -> {out_path}")
    
    print(f"\n{blindness_type.upper()} Correction Summary:")
    print(f"Processed: {processed} | Skipped: {skipped}")

def correct_all_types():
    """Generate corrections for all color blindness types"""
    input_dirs = {
        "protanopia": "data/simulated/protanopia/",
        "deuteranopia": "data/simulated/deuteranopia/",
        "tritanopia": "data/simulated/tritanopia/"
    }
    output_dirs = {
        "protanopia": "data/corrected/protanopia/",
        "deuteranopia": "data/corrected/deuteranopia/",
        "tritanopia": "data/corrected/tritanopia/"
    }
    
    print("Starting color correction process...")
    for cb_type in input_dirs:
        print(f"\nCorrecting {cb_type.upper()}...")
        process_folder(input_dirs[cb_type], output_dirs[cb_type], cb_type)
    print("\nAll corrections completed!")

if __name__ == "__main__":
    correct_all_types()