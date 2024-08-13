import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import logging
import matplotlib.pyplot as plt
from utils import NPZDataset, ToTensor, Normalize, Denormalize
from UNet import UNet

# Configure logging
logging.basicConfig(filename='log/runTest.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    filemode='w')

# Load test data
test_data_dir = "data/test/"
transform = transforms.Compose([ToTensor(), Normalize()])
test_dataset = NPZDataset(test_data_dir, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load the trained model
num_additional_inputs = 2
model = UNet(in_channels=2, out_channels=2, num_additional_inputs=num_additional_inputs)
model.load_state_dict(torch.load('UNet.pth'))
model.eval()  # Set the model to evaluation mode

with torch.no_grad():
    for i, batch in enumerate(test_dataloader):
        upper_z = batch['Upper_Z']
        lower_z = batch['Lower_Z']
        inputs = torch.cat((upper_z, lower_z), dim=1)  # Combine "Upper_Z" and "Lower_Z" as inputs
        upper_p = batch['Upper_P']
        lower_p = batch['Lower_P']
        targets = torch.cat((upper_p, lower_p), dim=1)  # Combine "Upper_P" and "Lower_P" as targets
        mach = batch['Mach'].unsqueeze(1)
        aoa = batch['AOA'].unsqueeze(1)
        fileName = batch['baseName'][0]
        baseName = fileName.split(".npz")[0]
        
        # Load normalization factors
        mean_upper_p = batch['mean_upper_p']
        std_upper_p = batch['std_upper_p']
        mean_lower_p = batch['mean_lower_p']
        std_lower_p = batch['std_lower_p']
        
        # Forward pass
        outputs = model(inputs, mach, aoa)
        
        # Split outputs back into Upper_P and Lower_P
        upper_p_output = outputs[:, 0, :, :].unsqueeze(1)
        lower_p_output = outputs[:, 1, :, :].unsqueeze(1)

        # Denormalize predictions
        denormalize = Denormalize(mean_upper_p, std_upper_p, mean_lower_p, std_lower_p)
        denorm_outputs = denormalize({'Upper_P': upper_p_output, 'Lower_P': lower_p_output})
        
        # Process Upper_P output for visualization
        upper_p_image = denorm_outputs['Upper_P'].squeeze().cpu().numpy()
        upper_p_image = np.flipud(upper_p_image.transpose())
        
        plt.figure(figsize=(8,6))
        plt.imshow(upper_p_image, cmap='jet', interpolation='nearest')
        plt.colorbar(label='Pressure')
        plt.title(f"Upper_P Prediction for Sample: {baseName}")
        plt.xlabel('X pixel')
        plt.ylabel('Y pixel')
        plt.savefig(f"plots/upper_p_prediction_{baseName}.png")
        plt.close()

        # Process Lower_P output for visualization
        lower_p_image = denorm_outputs['Lower_P'].squeeze().cpu().numpy()
        lower_p_image = np.flipud(lower_p_image.transpose())
        
        plt.figure(figsize=(8,6))
        plt.imshow(lower_p_image, cmap='jet', interpolation='nearest')
        plt.colorbar(label='Pressure')
        plt.title(f"Lower_P Prediction for Sample: {baseName}")
        plt.xlabel('X pixel')
        plt.ylabel('Y pixel')
        plt.savefig(f"plots/lower_p_prediction_{baseName}.png")
        plt.close()

        logging.info(f"Prediction images for sample {baseName} saved.")
