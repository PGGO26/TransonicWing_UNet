import logging
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import NPZDataset, Normalize, ToTensor
from UNet import UNet



# Configure logging
logging.basicConfig(filename='log/runTrain.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    filemode='w')

# Load training data
train_data_dir = "data/train/"
logging.info("Loading training data.")
transform = transforms.Compose([ToTensor(), Normalize()])
train_dataset = NPZDataset(train_data_dir, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Load validation data
val_data_dir = "data/validation/"
logging.info("Loading validation data.")
val_dataset = NPZDataset(val_data_dir, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Number of additional variables (Mach and AOA)
num_additional_inputs = 2

# Initialize model, loss function, and optimizer
model = UNet(in_channels=1, out_channels=1, num_additional_inputs =num_additional_inputs)
# model.apply(weights_init)
# if len(doLoad)>0:
#     model.load_state_dict(torch.load(doLoad))
#     print("Loaded model "+doLoad)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
num_epochs = 100
logging.info("Start training.")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for batch in train_dataloader:
        inputs = batch['Upper_Z']
        targets = batch['Upper_p']
        mach = batch['Mach'].unsqueeze(1)
        aoa = batch['AOA'].unsqueeze(1)

        # Forward pass
        outputs = model(inputs, mach, aoa)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    logging.info(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss/len(train_dataloader)}")

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            inputs = batch['Upper_Z']
            targets = batch['Upper_p']
            mach = batch['Mach'].unsqueeze(1)
            aoa = batch['AOA'].unsqueeze(1)

            outputs = model(inputs, mach, aoa)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    
    logging.info(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss/len(val_dataloader)}")

# Save the model
torch.save(model.state_dict(), 'UNet.pth')
logging.info("Model saved.")
