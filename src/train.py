import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import time
from torch.cuda.amp import autocast, GradScaler
import warnings

warnings.filterwarnings("ignore")

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

def load_data():
    X_train = np.load("data/processed/X_train.npy", mmap_mode='r')
    y_train = np.load("data/processed/y_train.npy", mmap_mode='r')
    types_train = np.load("data/processed/types_train.npy", mmap_mode='r')
    
    X_val = np.load("data/processed/X_val.npy", mmap_mode='r')
    y_val = np.load("data/processed/y_val.npy", mmap_mode='r')
    types_val = np.load("data/processed/types_val.npy", mmap_mode='r')
    
    def to_tensor(arr):
        return torch.FloatTensor(arr.transpose(0,3,1,2)/127.5 - 1.0)
    
    train_set = TensorDataset(
        to_tensor(X_train),
        to_tensor(y_train),
        torch.LongTensor(types_train)
    )
    
    val_set = TensorDataset(
        to_tensor(X_val),
        to_tensor(y_val),
        torch.LongTensor(types_val)
    )
    
    return train_set, val_set

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    train_set, val_set = load_data()
    batch_size = 16
    num_workers = min(4, os.cpu_count() - 1) if os.cpu_count() else 0
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                          num_workers=num_workers, pin_memory=True)
    
    model = ColorCorrectionNet(num_types=3).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    
    for epoch in range(20):
        model.train()
        train_loss = 0.0
        epoch_start = time.time()
        
        for inputs, targets, types in train_loader:
            inputs, targets, types = inputs.to(device), targets.to(device), types.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, types)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, types in val_loader:
                inputs, targets, types = inputs.to(device), targets.to(device), types.to(device)
                outputs = model(inputs, types)
                val_loss += criterion(outputs, targets).item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/color_correction.pth")
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1:02d}/20 | Time: {epoch_time:.1f}s | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train()