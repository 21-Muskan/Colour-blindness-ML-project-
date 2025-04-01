import torch
import torch.nn as nn
import cv2
import numpy as np
import os

class ColorCorrectionNet(nn.Module):
    def __init__(self, num_types=3):
        super().__init__()
        self.num_types = num_types
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3+num_types, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=1)
        )
        
        # Upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x, cb_type):
        # Create type embedding
        type_embed = torch.zeros((x.size(0), self.num_types, x.size(2), x.size(3))).to(x.device)
        for i, t in enumerate(cb_type):
            type_embed[i,t] = 1
        
        # Concatenate input with type embedding
        x = torch.cat([x, type_embed], dim=1)
        
        # Encoder
        x = self.enc1(x)
        x = nn.MaxPool2d(2)(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.up(x)
        x = self.decoder(x)
        
        return torch.tanh(x)

def load_model(model_path="models/color_correction.pth"):
    """Load the trained model from disk"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ColorCorrectionNet(num_types=3).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as e:
        raise RuntimeError(f"Error loading model weights: {str(e)}")
    
    model.eval()
    return model, device

def correct_image(input_path, output_path, blindness_type="protanopia"):
    """
    Correct an image for a specific color blindness type
    Args:
        input_path: Path to input image
        output_path: Path to save corrected image
        blindness_type: One of 'protanopia', 'deuteranopia', or 'tritanopia'
    """
    # Validate input type
    type_map = {"protanopia": 0, "deuteranopia": 1, "tritanopia": 2}
    if blindness_type.lower() not in type_map:
        raise ValueError(f"Invalid blindness type. Must be one of: {list(type_map.keys())}")
    
    type_idx = type_map[blindness_type.lower()]
    
    # Load model
    try:
        model, device = load_model()
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")
    
    # Read and validate input image
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found at {input_path}")
    
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not read image from {input_path}")
    
    original_size = img.shape[:2]
    
    try:
        # Preprocess image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        
        # Convert to tensor
        img_tensor = torch.FloatTensor(img.transpose(2,0,1)/127.5 - 1.0)
        img_tensor = img_tensor.unsqueeze(0).to(device)
        type_tensor = torch.LongTensor([type_idx]).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(img_tensor, type_tensor).cpu().numpy()
        
        # Postprocess
        output = (output[0].transpose(1,2,0) + 1) * 127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
        output = cv2.resize(output, (original_size[1], original_size[0]))
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, output)
        
    except Exception as e:
        raise RuntimeError(f"Error during image processing: {str(e)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python predict.py <input_path> <output_path> <blindness_type>")
        print("Blindness types: protanopia, deuteranopia, tritanopia")
        sys.exit(1)
    
    try:
        correct_image(sys.argv[1], sys.argv[2], sys.argv[3])
        print(f"Successfully processed image. Output saved to {sys.argv[2]}")
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)