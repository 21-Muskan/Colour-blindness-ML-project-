import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from typing import Tuple

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(in_channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.in2(out)
        out += residual
        return self.relu(out)

class ColorCorrectionNet(nn.Module):
    def __init__(self, num_types=3):
        super().__init__()
        self.num_types = num_types
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(4)]  # Must match train.py
        )
        
        # Type embedding
        self.type_embed = nn.Embedding(num_types, 256)
        
        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final convolution
        self.final_conv = nn.Conv2d(128, 3, kernel_size=7, padding=3)
        
    def forward(self, x, cb_type):
        # Initial features
        x = self.init_conv(x)
        
        # Downsample
        d1 = self.down1(x)
        d2 = self.down2(d1)
        
        # Residual blocks
        features = self.res_blocks(d2)
        
        # Type embedding
        type_embed = self.type_embed(cb_type)
        type_embed = type_embed.view(-1, 256, 1, 1).expand(-1, -1, features.size(2), features.size(3))
        
        # Concatenate features with type embedding
        combined = torch.cat([features, type_embed], dim=1)
        
        # Upsample
        u1 = self.up1(combined)
        u2 = self.up2(torch.cat([u1, d1], dim=1))
        
        # Final output
        output = self.final_conv(torch.cat([u2, x], dim=1))
        return torch.tanh(output)

def load_model(model_path="models/best_model.pth"):
    """Load trained model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ColorCorrectionNet(num_types=3).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as e:
        raise RuntimeError(f"Error loading weights: {str(e)}")
    
    model.eval()
    return model, device

def correct_image(input_path, output_path, blindness_type="protanopia"):
    """Correct image for color blindness"""
    type_map = {"protanopia": 0, "deuteranopia": 1, "tritanopia": 2}
    if blindness_type.lower() not in type_map:
        raise ValueError(f"Invalid type. Use: {list(type_map.keys())}")
    
    model, device = load_model()
    
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not read image: {input_path}")
    
    original_size = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))  # Match training size
    
    img_tensor = torch.FloatTensor(img.transpose(2,0,1)/127.5 - 1.0).unsqueeze(0).to(device)
    type_idx = torch.LongTensor([type_map[blindness_type.lower()]]).to(device)
    
    with torch.no_grad():
        output = model(img_tensor, type_idx).cpu().numpy()[0].transpose(1,2,0)
        output = (output + 1) * 127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
        output = cv2.resize(output, (original_size[1], original_size[0]))
        cv2.imwrite(output_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python predict.py <input.jpg> <output.jpg> <blindness_type>")
        sys.exit(1)
    
    try:
        correct_image(sys.argv[1], sys.argv[2], sys.argv[3])
        print(f"Success! Corrected image saved to {sys.argv[2]}")
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)