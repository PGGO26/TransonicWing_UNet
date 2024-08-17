import torch
import torch.nn as nn
import torch.nn.functional as F

class DFUNet(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, num_additional_inputs):
        super(DFUNet, self).__init__()
        
        # Local Feature 下采样方式 (带 Skip Connections)
        self.enc1_1 = self.conv_block(in_channels1, 32)
        self.enc2_1 = self.conv_block(32, 64)
        self.enc3_1 = self.conv_block(64, 128)
        self.enc4_1 = self.conv_block(128, 256)
        
        # Global Feature 下采样方式 (无 Skip Connections)
        self.enc1_2 = self.conv_block(in_channels2, 32)
        self.enc2_2 = self.conv_block(32, 64)
        self.enc3_2 = self.conv_block(64, 128)
        self.enc4_2 = self.conv_block(128, 256)
        
        # 頸部層带额外的变量
        self.bottom = self.conv_block(512 + num_additional_inputs, 512)
        
        # 上采样路径 (只使用 Local Feature 路径的 skip connections)
        self.upconv4 = self.upconv(512, 256, 2, 2)
        self.dec4 = self.conv_block(512, 256)
        
        self.upconv3 = self.upconv(256, 128, 2, 2)
        self.dec3 = self.conv_block(256, 128)
        
        self.upconv2 = self.upconv(128, 64, 2, 2)
        self.dec2 = self.conv_block(128, 64)
        
        self.upconv1 = self.upconv(64, 32, 3, 3)
        self.dec1 = self.conv_block(64, 32)
        
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def upconv(self, in_channels, out_channels, num_kernel_size, num_stride):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=num_kernel_size, stride=num_stride)
    
    def forward(self, local_features, global_features, mach, aoa, section):
        # Local Feature 下采样 (带 Skip Connections)
        enc1_1 = self.enc1_1(local_features)
        enc2_1 = self.enc2_1(F.max_pool2d(enc1_1, 3))
        enc3_1 = self.enc3_1(F.max_pool2d(enc2_1, 2))
        enc4_1 = self.enc4_1(F.max_pool2d(enc3_1, 2))
        
        # Global Feature 下采样 (无 Skip Connections)
        enc1_2 = self.enc1_2(global_features)
        enc2_2 = self.enc2_2(F.max_pool2d(enc1_2, 4))
        enc3_2 = self.enc3_2(F.max_pool2d(enc2_2, 4))
        enc4_2 = self.enc4_2(F.max_pool2d(enc3_2, 4))

        # 頸部層带额外的变量
        additional_inputs = torch.cat((mach, aoa, section), dim=1).unsqueeze(2).unsqueeze(3).expand(-1, -1, enc4_1.shape[2] // 2, enc4_1.shape[3] // 2)
        print("Additional inputs shape: ", additional_inputs.size())  # 调试打印
        bottom_input = torch.cat((F.max_pool2d(enc4_1, 2), F.max_pool2d(enc4_2, 2), additional_inputs), dim=1)
        bottom = self.bottom(bottom_input)

        # 上采样路径
        dec4 = self.upconv4(bottom)
        dec4 = torch.cat((enc4_1, dec4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((enc3_1, dec3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2_1, dec2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1_1, dec1), dim=1)
        dec1 = self.dec1(dec1)
        
        return self.final_conv(dec1)
