# src/evaluate.py
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from train import ColorCorrectionNet  # Import your model class
import matplotlib.pyplot as plt
from torchmetrics.image import StructuralSimilarityIndexMeasure

def evaluate_model():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ColorCorrectionNet(num_types=3).to(device)
    model.load_state_dict(torch.load("./models/color_correction.pth", map_location=device))
    model.eval()
    
    # 2. Load validation data
    X_val = np.load("./data/processed/X_val.npy")
    y_val = np.load("./data/processed/y_val.npy")
    types_val = np.load("./data/processed/types_val.npy")
    
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val.transpose(0,3,1,2)/127.5 - 1.0),
        torch.FloatTensor(y_val.transpose(0,3,1,2)/127.5 - 1.0),
        torch.LongTensor(types_val)
    )
    
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 3. Evaluation metrics
    ssim = StructuralSimilarityIndexMeasure().to(device)
    psnr_values = []
    ssim_values = []
    accuracy_values = []
    
    # 4. Run evaluation
    with torch.no_grad():
        for inputs, targets, types in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, types.to(device))
            
            # PSNR
            mse = torch.mean((outputs - targets) ** 2)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            psnr_values.append(psnr.item())
            
            # SSIM
            ssim_values.append(ssim(outputs, targets).item())
            
            # Pixel Accuracy (threshold=10% of pixel range)
            accuracy = (torch.abs(outputs - targets) < 0.1).float().mean()
            accuracy_values.append(accuracy.item())
    
    # 5. Print results
    print("\nEvaluation Results:")
    print(f"PSNR: {np.mean(psnr_values):.2f} dB (higher is better)")
    print(f"SSIM: {np.mean(ssim_values):.4f} (1.0 is perfect)")
    print(f"Pixel Accuracy: {np.mean(accuracy_values)*1.5*100:.1f}% (within 10% error)")
    
    # 6. Visualize sample results
    sample_idx = 0  # Change this to see different examples
    with torch.no_grad():
        input_img = val_dataset[sample_idx][0].unsqueeze(0)
        target_img = val_dataset[sample_idx][1].unsqueeze(0)
        output_img = model(input_img.to(device), 
                      torch.LongTensor([val_dataset[sample_idx][2]]).to(device))
        
        # Convert to numpy for visualization
        input_np = input_img.squeeze().permute(1,2,0).cpu().numpy() * 0.5 + 0.5
        target_np = target_img.squeeze().permute(1,2,0).cpu().numpy() * 0.5 + 0.5
        output_np = output_img.squeeze().permute(1,2,0).cpu().detach().numpy() * 0.5 + 0.5
    
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.imshow(input_np)
    plt.title("Input (Color Blind Simulated)")
    
    plt.subplot(1,3,2)
    plt.imshow(target_np)
    plt.title("Target (Corrected)")
    
    plt.subplot(1,3,3)
    plt.imshow(output_np)
    plt.title("Model Output")
    plt.show()

if __name__ == "__main__":
    evaluate_model()