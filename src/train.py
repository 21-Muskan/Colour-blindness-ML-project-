import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import cv2
import gc
import psutil
import warnings
warnings.filterwarnings("ignore")

# Configuration
CONFIG = {
    'batch_size': 4,
    'num_epochs': 50,
    'learning_rate': 0.0002,
    'image_size': (128, 128),
    'num_workers': 0,
    'save_interval': 1,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

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
            *[ResidualBlock(256) for _ in range(4)]  # Reduced from 6 to 4 for memory
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

def load_and_resize_data(data_dir):
    """Load and resize data with memory efficiency"""
    print("Loading and resizing data...")
    
    data_files = {
        'train': ('X_train.npy', 'y_train.npy', 'types_train.npy'),
        'val': ('X_val.npy', 'y_val.npy', 'types_val.npy')
    }
    
    loaded_data = {}
    for split, (x_file, y_file, t_file) in data_files.items():
        x_path = os.path.join(data_dir, x_file)
        y_path = os.path.join(data_dir, y_file)
        t_path = os.path.join(data_dir, t_file)
        
        if not all(os.path.exists(p) for p in [x_path, y_path, t_path]):
            raise FileNotFoundError(f"Missing data files in {data_dir}")
        
        loaded_data[f'X_{split}'] = np.array([cv2.resize(img, CONFIG['image_size']) 
                                           for img in np.load(x_path, mmap_mode='r')])
        loaded_data[f'y_{split}'] = np.array([cv2.resize(img, CONFIG['image_size']) 
                                           for img in np.load(y_path, mmap_mode='r')])
        loaded_data[f't_{split}'] = np.load(t_path)
    
    return loaded_data

def create_datasets(data):
    """Create datasets with memory-efficient conversion"""
    def to_tensor(arr):
        chunks = [arr[i:i+50] for i in range(0, arr.shape[0], 50)]
        tensors = [torch.FloatTensor(chunk.transpose(0,3,1,2)/127.5 - 1.0) for chunk in chunks]
        return torch.cat(tensors)
    
    train_set = TensorDataset(
        to_tensor(data['X_train']),
        to_tensor(data['y_train']),
        torch.LongTensor(data['t_train'])
    )
    
    val_set = TensorDataset(
        to_tensor(data['X_val']),
        to_tensor(data['y_val']),
        torch.LongTensor(data['t_val'])
    )
    
    return train_set, val_set

def train_model():
    torch.manual_seed(42)
    print(f"\nUsing device: {CONFIG['device']}")
    
    try:
        data = load_and_resize_data("data/processed")
        train_set, val_set = create_datasets(data)
        del data
        gc.collect()
        
        train_loader = DataLoader(
            train_set,
            batch_size=CONFIG['batch_size'],
            shuffle=True,
            num_workers=CONFIG['num_workers'],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_set,
            batch_size=CONFIG['batch_size'],
            num_workers=CONFIG['num_workers'],
            pin_memory=True
        )
        
        model = ColorCorrectionNet(num_types=3).to(CONFIG['device'])
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
        criterion = nn.L1Loss()
        
        best_val_loss = float('inf')
        
        for epoch in range(CONFIG['num_epochs']):
            model.train()
            train_loss = 0.0
            
            for batch_idx, (inputs, targets, types) in enumerate(train_loader):
                inputs, targets, types = [t.to(CONFIG['device']) for t in [inputs, targets, types]]
                
                optimizer.zero_grad()
                outputs = model(inputs, types)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 20 == 0:
                    print(f"\rEpoch {epoch+1}/{CONFIG['num_epochs']} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}", end="")
                    gc.collect()
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets, types in val_loader:
                    inputs, targets, types = [t.to(CONFIG['device']) for t in [inputs, targets, types]]
                    outputs = model(inputs, types)
                    val_loss += criterion(outputs, targets).item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "models/best_model.pth")
            
            print(f"\nEpoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")
        
        print("\nTraining completed!")
        
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_model()