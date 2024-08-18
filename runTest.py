import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from utils import NPZDataset, ToTensor, Normalize, Denormalize
from networks.UNet import UNet

def load_test_data(test_data_dir):
    transform = transforms.Compose([ToTensor(), Normalize()])
    test_dataset = NPZDataset(test_data_dir, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return test_dataloader

def load_model(model_path, num_additional_inputs=2):
    model = UNet(in_channels=2, out_channels=2, num_additional_inputs=num_additional_inputs)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def denormalize_outputs(mean_upper_p, std_upper_p, mean_lower_p, std_lower_p, upper_p_output, lower_p_output, upper_p, lower_p):
    denormalize = Denormalize(mean_upper_p, std_upper_p, mean_lower_p, std_lower_p)
    denorm_outputs = denormalize({'Upper_P': upper_p_output, 'Lower_P': lower_p_output})
    denorm_targets = denormalize({'Upper_P': upper_p, 'Lower_P': lower_p})
    return denorm_outputs, denorm_targets

def calculate_and_visualize_error(predict, target, baseName, region, cmap='Greys'):
    error = np.abs(predict - target)
    error_image = np.flipud(error.transpose())
    plt.figure(figsize=(8, 6))
    plt.imshow(error_image, cmap=cmap, interpolation='nearest')
    plt.colorbar(label='Prediction Error')
    plt.title(f"{region} Error for Sample: {baseName}")
    plt.xlabel('X pixel')
    plt.ylabel('Y pixel')
    plt.savefig(f"plots/{region.lower()}_error_{baseName}.png")
    plt.close()
    print(f"{region} error : {np.mean(error)}\n{region} Max error : {np.max(error)}")
    return error_image

def visualize_prediction(predict, baseName, region, cmap='jet'):
    predict_image = np.flipud(predict.transpose())
    plt.figure(figsize=(8, 6))
    plt.imshow(predict_image, cmap=cmap, interpolation='nearest')
    plt.colorbar(label='Pressure')
    plt.title(f"{region} Prediction for Sample: {baseName}")
    plt.xlabel('X pixel')
    plt.ylabel('Y pixel')
    plt.savefig(f'plots/{region.lower()}_prediction_{baseName}.png')
    plt.close()

def visualize_ground_truth(target, baseName, region, cmap='jet'):
    target_image = np.flipud(target.transpose())
    plt.figure(figsize=(8, 6))
    plt.imshow(target_image, cmap=cmap, interpolation='nearest')
    plt.colorbar(label='Pressure')
    plt.title(f"{region} Ground Truth for Sample: {baseName}")
    plt.xlabel('X pixel')
    plt.ylabel('Y pixel')
    plt.savefig(f'plots/{region.lower()}_ground_{baseName}.png')
    plt.close()

def main():
    test_data_dir = "data/test/"
    model_path = 'models/UNet.pth'

    # Load test data and model
    test_dataloader = load_test_data(test_data_dir)
    model = load_model(model_path)

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            upper_z = batch['Upper_Z']
            lower_z = batch['Lower_Z']
            inputs = torch.cat((upper_z, lower_z), dim=1)
            upper_p = batch['Upper_P']
            lower_p = batch['Lower_P']
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
            upper_p_output = outputs[:, 0, :, :].unsqueeze(1)
            lower_p_output = outputs[:, 1, :, :].unsqueeze(1)

            # Denormalize outputs and targets
            denorm_outputs, denorm_targets = denormalize_outputs(
                mean_upper_p, std_upper_p, mean_lower_p, std_lower_p,
                upper_p_output, lower_p_output, upper_p, lower_p
            )

            # Calculate errors and generate plots for Upper_P
            upper_p_predict = denorm_outputs['Upper_P'].squeeze().cpu().numpy()
            upper_p_target = denorm_targets['Upper_P'].squeeze().cpu().numpy()
            calculate_and_visualize_error(upper_p_predict, upper_p_target, baseName, "Upper_P")
            visualize_prediction(upper_p_predict, baseName, "Upper_P")
            visualize_ground_truth(upper_p_target, baseName, "Upper_P")

            # Calculate errors and generate plots for Lower_P
            lower_p_predict = denorm_outputs['Lower_P'].squeeze().cpu().numpy()
            lower_p_target = denorm_targets['Lower_P'].squeeze().cpu().numpy()
            calculate_and_visualize_error(lower_p_predict, lower_p_target, baseName, "Lower_P")
            visualize_prediction(lower_p_predict, baseName, "Lower_P")
            visualize_ground_truth(lower_p_target, baseName, "Lower_P")

            print(f"Prediction and error images for sample {baseName} saved.")

if __name__ == "__main__":
    main()
