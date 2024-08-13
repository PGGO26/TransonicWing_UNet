import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_additional_inputs):
        super(UNet, self).__init__()
        
        # Contracting path (Encoder)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottom layer with additional variables
        self.bottom = self.conv_block(512 + num_additional_inputs, 1024)
        
        # Expansive path (Decoder)
        self.upconv4 = self.upconv(1024, 512)
        self.dec4 = self.conv_block(1024, 512)
        
        self.upconv3 = self.upconv(512, 256)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = self.upconv(256, 128)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = self.upconv(128, 64)
        self.dec1 = self.conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x, mach, aoa):
        # Contracting path
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        
        # Bottom layer with additional variables
        additional_inputs = torch.cat((mach, aoa), dim=1).unsqueeze(2).unsqueeze(3).expand(-1, -1, enc4.shape[2] // 2, enc4.shape[3] // 2)
        bottom_input = torch.cat((F.max_pool2d(enc4, 2), additional_inputs), dim=1)
        bottom = self.bottom(bottom_input)
        
        # Expansive path
        dec4 = self.upconv4(bottom)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.dec1(dec1)
        
        return self.final_conv(dec1)
