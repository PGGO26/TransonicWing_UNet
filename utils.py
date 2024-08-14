import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import cv2

class NPZDataset(Dataset):
    def __init__(self, folder_path, transform=None, augmentations=None):
        self.folder_path = folder_path
        self.npz_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npz')]
        self.transform = transform
        self.augmentations = augmentations

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
            'Upper_P': upper_p,
            'Lower_Z': lower_z,
            'Lower_P': lower_P,
            'Mach': mach,
            'AOA': aoa,
            'baseName': filename
        }

        # Apply data augmentations if any
        if self.augmentations:
            sample = self.augmentations(sample)

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor:
    def __call__(self, sample):
        upper_z, upper_p, lower_z, lower_P = sample['Upper_Z'], sample['Upper_P'], sample['Lower_Z'], sample['Lower_P']
        mach, aoa = sample['Mach'], sample['AOA']
        fileName = sample['baseName']
        
        # Convert additional variables to tensor
        mach_tensor = torch.tensor(mach, dtype=torch.float32)
        aoa_tensor = torch.tensor(aoa, dtype=torch.float32)
        
        return {
            'Upper_Z': torch.tensor(upper_z, dtype=torch.float32).unsqueeze(0),
            'Upper_P': torch.tensor(upper_p, dtype=torch.float32).unsqueeze(0),
            'Lower_Z': torch.tensor(lower_z, dtype=torch.float32).unsqueeze(0),
            'Lower_P': torch.tensor(lower_P, dtype=torch.float32).unsqueeze(0),
            'Mach': mach_tensor,
            'AOA': aoa_tensor,
            'baseName': fileName
        }

class Normalize:
    def __call__(self, sample):
        upper_z, upper_p, lower_z, lower_P = sample['Upper_Z'], sample['Upper_P'], sample['Lower_Z'], sample['Lower_P']
        
        # Compute normalization statistics
        self.mean_upper_p = torch.mean(upper_p).item()
        self.std_upper_p = torch.std(upper_p).item()
        self.mean_lower_p = torch.mean(lower_P).item()
        self.std_lower_p = torch.std(lower_P).item()
        
        # Normalize input tensors
        norm_upper_z = (upper_z - torch.mean(upper_z)) / torch.std(upper_z)
        norm_upper_p = (upper_p - self.mean_upper_p) / self.std_upper_p
        norm_lower_z = (lower_z - torch.mean(lower_z)) / torch.std(lower_z)
        norm_lower_P = (lower_P - self.mean_lower_p) / self.std_lower_p

        return {
            'Upper_Z': norm_upper_z,
            'Upper_P': norm_upper_p,
            'Lower_Z': norm_lower_z,
            'Lower_P': norm_lower_P,
            'Mach': sample['Mach'],
            'AOA': sample['AOA'],
            'baseName': sample['baseName'],
            'mean_upper_p': self.mean_upper_p,
            'mean_lower_p': self.mean_lower_p,
            'std_upper_p': self.std_upper_p,
            'std_lower_p': self.std_lower_p
        }

class Denormalize:
    def __init__(self, mean_upper_p, std_upper_p, mean_lower_p, std_lower_p):
        self.mean_upper_p = mean_upper_p
        self.std_upper_p = std_upper_p
        self.mean_lower_p = mean_lower_p
        self.std_lower_p = std_lower_p

    def __call__(self, sample):
        upper_p, lower_p = sample['Upper_P'], sample['Lower_P']
        
        # Denormalize tensors
        denorm_upper_p = upper_p * self.std_upper_p + self.mean_upper_p
        denorm_lower_P = lower_p * self.std_lower_p + self.mean_lower_p

        return {
            'Upper_P': denorm_upper_p,
            'Lower_P': denorm_lower_P
        }

class GeometricTransformations:
    """Apply geometric transformations like rotation, translation, and scaling"""
    def __call__(self, sample):
        angle = random.uniform(-30, 30)  # Random rotation angle
        scale = random.uniform(0.8, 1.2)  # Random scale factor
        tx = random.uniform(-10, 10)  # Random translation in x direction
        ty = random.uniform(-10, 10)  # Random translation in y direction
        
        transform_matrix = cv2.getRotationMatrix2D((sample['Upper_Z'].shape[1] / 2, sample['Upper_Z'].shape[0] / 2), angle, scale)
        transform_matrix[:, 2] += [tx, ty]  # Add translation to transformation matrix
        
        for key in ['Upper_Z', 'Upper_P', 'Lower_Z', 'Lower_P']:
            sample[key] = cv2.warpAffine(sample[key], transform_matrix, (sample[key].shape[1], sample[key].shape[0]))
        
        return sample

class ApplyFilters:
    """Apply filters like GaussianBlur or Sobel"""
    def __call__(self, sample):
        if random.random() > 0.5:
            for key in ['Upper_Z', 'Upper_P', 'Lower_Z', 'Lower_P']:
                sample[key] = cv2.GaussianBlur(sample[key], (5, 5), 0)
        if random.random() > 0.5:
            for key in ['Upper_Z', 'Upper_P', 'Lower_Z', 'Lower_P']:
                sample[key] = cv2.Sobel(sample[key], cv2.CV_64F, 1, 0, ksize=5)
        
        return sample
