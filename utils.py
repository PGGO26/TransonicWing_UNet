import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

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
        lower_P = npz_data['Lower_P']

        # Extract Mach and AOA from the filename
        filename = os.path.basename(npz_file)
        parts = filename.split('_')
        mach = float(parts[-2])
        aoa = float(parts[-1].replace('.npz', ''))

        sample = {
            'Upper_Z': upper_z,
            'Upper_p': upper_p,
            'Lower_Z': lower_z,
            'Lower_P': lower_P,
            'Mach': mach,
            'AOA': aoa,
            'baseName' : filename
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor:
    def __call__(self, sample):
        upper_z, upper_p, lower_z, lower_P = sample['Upper_Z'], sample['Upper_p'], sample['Lower_Z'], sample['Lower_P']
        mach, aoa = sample['Mach'], sample['AOA']
        fileName = sample['baseName']
        
        # Convert additional variables to tensor
        mach_tensor = torch.tensor(mach, dtype=torch.float32)
        aoa_tensor = torch.tensor(aoa, dtype=torch.float32)
        
        return {
            'Upper_Z': torch.tensor(upper_z, dtype=torch.float32).unsqueeze(0),
            'Upper_p': torch.tensor(upper_p, dtype=torch.float32).unsqueeze(0),
            'Lower_Z': torch.tensor(lower_z, dtype=torch.float32).unsqueeze(0),
            'Lower_P': torch.tensor(lower_P, dtype=torch.float32).unsqueeze(0),
            'Mach': mach_tensor,
            'AOA': aoa_tensor,
            'baseName' : fileName
        }

class Normalize:
    def __call__(self, sample):
        upper_z, upper_p, lower_z, lower_P = sample['Upper_Z'], sample['Upper_p'], sample['Lower_Z'], sample['Lower_P']
        
        # Normalize input tensors
        norm_upper_z = (upper_z - torch.mean(upper_z)) / torch.std(upper_z)
        norm_upper_p = (upper_p - torch.mean(upper_p)) / torch.std(upper_p)
        norm_lower_z = (lower_z - torch.mean(lower_z)) / torch.std(lower_z)
        norm_lower_P = (lower_P - torch.mean(lower_P)) / torch.std(lower_P)

        return {
            'Upper_Z': norm_upper_z,
            'Upper_p': norm_upper_p,
            'Lower_Z': norm_lower_z,
            'Lower_P': norm_lower_P,
            'Mach': sample['Mach'],
            'AOA': sample['AOA'],
            'baseName' : sample['baseName']
        }
