import os
import pandas as pd
import numpy as np

class outputProcessing:

    ## Parameters
    modeUsage = ["boundaryCloud", "internalCloud"]
    boundary_postProcess = ["Upper","Lower"]
    Section_Lst = [0.2, 0.65, 0.8, 0.9, 0.96]
    postProcess_range = 1.8
    postProcess_range_sub = 1.5
    postProcess_x_min = -0.15
    postProcess_z_AIP_min = -0.9
    Output_channel_UNet = 4
    Output_channel_DF = 5
    Output_channel_DF_Section = 4

    def __init__(self, baseName, postProcessDir="OpenFOAM/postProcessing/", dataDir="train/", count=1, boundary_res=256, internal_res=1024, Vel=319.083):
        self.baseName = baseName
        self.postProcessDir = postProcessDir
        self.dataDir = dataDir
        self.count = count
        self.Vel = Vel
        self.boundary_res = boundary_res
        self.internal_res = internal_res

    def output_npz(self):
        self.Output_UNet = np.ones((self.Output_channel_UNet, self.boundary_res, self.boundary_res))
        self.Output_DF = np.zeros((self.Output_channel_DF, self.Output_channel_DF_Section, self.internal_res, self.internal_res))
        print(f"\tDoing postProcessing {self.modeUsage[0]}.")
        self.postProcess_Process(mode=self.modeUsage[0])
        print(f"\tDoing postProcessing {self.modeUsage[1]}.")
        self.postProcess_Process(mode=self.modeUsage[1])
        fileName = self.dataDir + self.baseName
        print("\tsaving in " + fileName + f".npz")
        np.savez_compressed(fileName, Upper_Z=self.Output_UNet[0],Upper_P=self.Output_UNet[1],Lower_Z=self.Output_UNet[2],Lower_P=self.Output_UNet[3],
                                  Local_20pc=self.Output_DF[0],Local_65pc=self.Output_DF[1],Local_80pc=self.Output_DF[2],
                                  Local_90pc=self.Output_DF[3],Local_96pc=self.Output_DF[4])

    def postProcess_Process(self, mode="boundaryCloud"):
        if mode not in self.modeUsage:
            raise ValueError(f"Invalid usage mode : {mode}, Available options are {modeUsage}.")
        
        dataDir = self.postProcessDir
        timestep = os.listdir(dataDir + mode + "/")[0]
        timestep_path = os.path.join(dataDir, mode, timestep)
        self.Current_Mode = mode
        for process_case in os.listdir(timestep_path):
            self.Case = process_case
            self.Case_Path = os.path.join(timestep_path, process_case)
            if mode == "boundaryCloud":
                self.postProcess_array(res=self.boundary_res, columns_Names=['x','y','z','p'])
            elif mode == "internalCloud":
                self.postProcess_array(res=self.internal_res, columns_Names=['x','y','z','p','u','v','w'])

    def postProcess_array(self, res, columns_Names):
        content = pd.read_csv(self.Case_Path, delimiter="\s+", skiprows=0, header=None, names=columns_Names)
        print(f"Loading data in {self.Case_Path}")
        x = content["x"]
        y = content["y"]
        z = content["z"]
        p = content["p"]

        if self.Current_Mode == "boundaryCloud":
            normalized_x = (x - self.postProcess_x_min) / self.postProcess_range
            normalized_y = y / self.postProcess_range_sub
            if os.path.splitext(self.Case)[0] == self.boundary_postProcess[0]:
                upper_z = np.ones((res, res))
                upper_p = np.ones((res, res))
                upper_z[(normalized_x * (res - 1)).astype(int), (normalized_y * (res - 1)).astype(int)] = (z - z.min()) / (z.max() - z.min())
                upper_p[(normalized_x * (res - 1)).astype(int), (normalized_y * (res - 1)).astype(int)] = p
                self.Output_UNet[0] = upper_z
                self.Output_UNet[1] = upper_p
                print(f"OutputProcess ({self.boundary_postProcess[0]} done.)")
            elif os.path.splitext(self.Case)[0] == self.boundary_postProcess[1]:
                lower_z = np.zeros((res, res))
                lower_p = np.zeros((res, res))
                lower_z[(normalized_x * (res - 1)).astype(int), (normalized_y * (res - 1)).astype(int)] = (z - z.min()) / (z.max() - z.min())
                lower_p[(normalized_x * (res - 1)).astype(int), (normalized_y * (res - 1)).astype(int)] = p
                self.Output_UNet[2] = lower_z
                self.Output_UNet[3] = lower_p
                print(f"OutputProcess ({self.boundary_postProcess[1]} done.)")

        elif self.Current_Mode == "internalCloud":
            index = 0
            for section in self.Section_Lst:
                print(f"Section {section} postProcessing.")
                section_df = content[(content['y'] >= section * 1.1963 - 1e-2) & (content['y'] <= section * 1.1963 + 1e-2)]
                ix = section_df['x']
                iz = section_df['z']
                normalized_ix = (ix - self.postProcess_x_min) / self.postProcess_range
                normalized_iz = (iz - (self.postProcess_z_AIP_min)) / self.postProcess_range
                self.Output_DF[index][0][(normalized_ix * (res-1)).astype(int), (normalized_iz * (res-1)).astype(int)] = section_df['p']
                self.Output_DF[index][1][(normalized_ix * (res-1)).astype(int), (normalized_iz * (res-1)).astype(int)] = section_df['u']
                self.Output_DF[index][2][(normalized_ix * (res-1)).astype(int), (normalized_iz * (res-1)).astype(int)] = section_df['v']
                self.Output_DF[index][3][(normalized_ix * (res-1)).astype(int), (normalized_iz * (res-1)).astype(int)] = section_df['w']
                index += 1
