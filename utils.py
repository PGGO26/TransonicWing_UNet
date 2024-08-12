import os
import numpy as np
import torch
from torch.utils.data import Dataset

class NPZDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.npz_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npz')]
        self.transform = transform

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        npz_file = self.npz_files[idx]
        npz_data = np.load(npz_file)
        upper_z = npz_data['Upper_Z']
        upper_p = npz_data['Upper_P']
        lower_z = npz_data['Lower_Z']
        lower_p = npz_data['Lower_P']

        # Extract Mach and AOA from the filename
        filename = os.path.basename(npz_file)
        parts = filename.split('_')
        mach = float(parts[-2])
        aoa = float(parts[-1].replace('.npz', ''))

        sample = {
            'Upper_Z': upper_z,
            'Upper_P': upper_p,
            'Lower_Z': lower_z,
            'Lower_P': lower_p,
            'Mach': mach,
            'AOA': aoa,
            'baseName': filename
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor:
    def __call__(self, sample):
        upper_z, upper_p, lower_z, lower_p = sample['Upper_Z'], sample['Upper_P'], sample['Lower_Z'], sample['Lower_P']
        mach, aoa = sample['Mach'], sample['AOA']
        filename = sample['baseName']
        
        # Convert additional variables to tensor
        mach_tensor = torch.tensor(mach, dtype=torch.float32)
        aoa_tensor = torch.tensor(aoa, dtype=torch.float32)
        
        # Stack inputs and targets to create multi-channel tensors
        inputs = torch.stack([torch.tensor(upper_z, dtype=torch.float32), torch.tensor(lower_z, dtype=torch.float32)], dim=0)
        targets = torch.stack([torch.tensor(upper_p, dtype=torch.float32), torch.tensor(lower_p, dtype=torch.float32)], dim=0)
        
        return {
            'inputs': inputs,
            'targets': targets,
            'Mach': mach_tensor,
            'AOA': aoa_tensor,
            'baseName': filename
        }

class Normalize:
    def __call__(self, sample):
        inputs, targets = sample['inputs'], sample['targets']
        
        # Calculate mean and std for each channel
        mean_upper_z, std_upper_z = torch.mean(inputs[0]), torch.std(inputs[0])
        mean_lower_z, std_lower_z = torch.mean(inputs[1]), torch.std(inputs[1])
        mean_upper_p, std_upper_p = torch.mean(targets[0]), torch.std(targets[0])
        mean_lower_p, std_lower_p = torch.mean(targets[1]), torch.std(targets[1])
        
        # Normalize each channel
        inputs[0] = (inputs[0] - mean_upper_z) / std_upper_z
        inputs[1] = (inputs[1] - mean_lower_z) / std_lower_z
        targets[0] = (targets[0] - mean_upper_p) / std_upper_p
        targets[1] = (targets[1] - mean_lower_p) / std_lower_p

        return {
            'inputs': inputs,
            'targets': targets,
            'Mach': sample['Mach'],
            'AOA': sample['AOA'],
            'baseName': sample['baseName'],
            'Mean_Std': {
                'mean_upper_z': mean_upper_z,
                'std_upper_z': std_upper_z,
                'mean_lower_z': mean_lower_z,
                'std_lower_z': std_lower_z,
                'mean_upper_p': mean_upper_p,
                'std_upper_p': std_upper_p,
                'mean_lower_p': mean_lower_p,
                'std_lower_p': std_lower_p
            }
        }

class Denormalize:
    def __call__(self, sample):
        mean_std = sample['Mean_Std']
        mean_upper_z, std_upper_z = mean_std['mean_upper_z'], mean_std['std_upper_z']
        mean_lower_z, std_lower_z = mean_std['mean_lower_z'], mean_std['std_lower_z']
        mean_upper_p, std_upper_p = mean_std['mean_upper_p'], mean_std['std_upper_p']
        mean_lower_p, std_lower_p = mean_std['mean_lower_p'], mean_std['std_lower_p']
        
        inputs, targets = sample['inputs'], sample['targets']
        
        # Denormalize each channel
        inputs[0] = inputs[0] * std_upper_z + mean_upper_z
        inputs[1] = inputs[1] * std_lower_z + mean_lower_z
        targets[0] = targets[0] * std_upper_p + mean_upper_p
        targets[1] = targets[1] * std_lower_p + mean_lower_p

        return {
            'inputs': inputs,
            'targets': targets,
            'Mach': sample['Mach'],
            'AOA': sample['AOA'],
            'baseName': sample['baseName']
        }

def load_data(folder_path, batch_size=32):
    transform = transforms.Compose([ToTensor(), Normalize()])
    dataset = NPZDataset(folder_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
