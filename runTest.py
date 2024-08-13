import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import logging
import matplotlib.pyplot as plt
from utils import NPZDataset, ToTensor, Normalize
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
model = UNet(in_channels=1, out_channels=1, num_additional_inputs=num_additional_inputs)
model.load_state_dict(torch.load('UNet.pth'))
model.eval()  # Set the model to evaluation mode

with torch.no_grad():
    for i, batch in enumerate(test_dataloader):
        inputs = batch['Upper_Z']
        targets = batch['Upper_p']
        mach = batch['Mach'].unsqueeze(1)
        aoa = batch['AOA'].unsqueeze(1)
        fileName = batch['baseName'][0]
        baseName = fileName.split(".npz")[0]
        
        # Forward pass
        outputs = model(inputs, mach, aoa)
        
        # Process and save prediction
        output_image = outputs.squeeze().cpu().numpy()
        output_image = np.flipud(output_image.transpose())
        
        plt.figure(figsize=(8,6))
        plt.imshow(output_image, cmap='jet', interpolation='nearest')
        plt.colorbar(label='Pressure')
        plt.title(f"Prediction for Sample : {baseName}")
        plt.xlabel('X pixel')
        plt.ylabel('Y pixel')
        plt.savefig(f"plots/prediction_{baseName}.png")
        plt.close()

        logging.info(f"Prediction image for sample {baseName} saved.")
