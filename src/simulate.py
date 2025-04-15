import cv2
import numpy as np

def simulate_color_blindness(input_path, output_path, blindness_type="protanopia"):
    """Simulate color blindness on an image"""
    # Color transformation matrices for different types of color blindness
    matrices = {
        "protanopia": np.array([
            [0.567, 0.433, 0],
            [0.558, 0.442, 0],
            [0, 0.242, 0.758]
        ]),
        "deuteranopia": np.array([
            [0.625, 0.375, 0],
            [0.7, 0.3, 0],
            [0, 0.3, 0.7]
        ]),
        "tritanopia": np.array([
            [0.95, 0.05, 0],
            [0, 0.433, 0.567],
            [0, 0.475, 0.525]
        ])
    }
    
    if blindness_type.lower() not in matrices:
        raise ValueError(f"Invalid type. Use: {list(matrices.keys())}")
    
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not read image: {input_path}")
    
    # Convert to RGB and apply transformation
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    simulated = cv2.transform(img_rgb, matrices[blindness_type.lower()])
    
    # Convert back to BGR and save
    simulated_bgr = cv2.cvtColor(simulated, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, simulated_bgr)