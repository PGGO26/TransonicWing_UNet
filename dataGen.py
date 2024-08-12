import os
import shutil
import re
import numpy as np
import utils_dataGen

outputDir = 'train/'
MESH_DIR = 'Mesh/'
V_sonic = 319.083
Mach_values = [0.9, 0.85, 0.8, 0.75]
angle_values = list(range(6, 16))
sample = len(Mach_values) * len(angle_values) * len(os.listdir(MESH_DIR))

print(f"Number of samples: {sample}")

def get_numeric_files(filelst):
    numeric_files = []
    for file in filelst:
        try:
            num_file = float(file)
            numeric_files.append(num_file)
        except ValueError:
            continue
    return numeric_files

def modify_files(freestream_x, freestream_z):
    # Backup and modify the boundary file
    shutil.copy("constant/polyMesh/boundary", "constant/boundary_Template")
    with open("constant/boundary_Template", "rt") as infile:
        content = infile.read()
    content = re.sub(r'(symm\s*\{[^}]*type\s*wall;)', r'\1\n    type            symmetry;', content)
    content = re.sub(r'(farfield\s*\{[^}]*type\s*wall;)', r'\1\n    type            farfield;', content)
    with open("constant/polyMesh/boundary", "wt") as outfile:
        outfile.write(content)

    # Modify freestreamConditions file
    with open("0/include/template", "rt") as infile:
        with open("0/include/freestreamConditions", "wt") as outfile:
            for line in infile:
                line = line.replace("VEL_X", f"{freestream_x}")
                line = line.replace("VEL_Z", f"{freestream_z}")
                outfile.write(line)

def run_simulation(freestream_x, freestream_z):
    modify_files(freestream_x, freestream_z)

    # Check and run decomposePar if necessary
    processor_folders = [f'processor{i}' for i in range(8)]
    if all(os.path.isdir(folder) for folder in processor_folders):
        print("All processor folders exist. Skipping decomposePar.")
    else:
        print("Not all processor folders exist. Running decomposePar.")
        os.system("decomposePar > log/decomposePar.log")

    # Run HiSa with error checking
    print("Running HiSa.")
    hisa_result = os.system('mpirun -n 8 hisa -parallel > log/hisa.log')
    if hisa_result != 0:
        print("HiSa simulation failed. Skipping to next Mach and angle.")
        return False  # Return False if the simulation fails

    print("Simulation done.")
    os.system("reconstructPar > log/reconstructPar.log")
    return True  # Return True if the simulation succeeds


def post_process(path):
    postProcessDir = "postProcessing/"
    print(f"Post-processing case: {path}")

    boundary_path = os.path.join(postProcessDir, "boundaryCloud")
    # Post-process Upper Surface
    print("Post-processing Upper Surface.")
    with open("system/boundaryCloud_Template", "rt") as inFile:
        with open("system/boundaryCloud", "wt") as outFile:
            for line in inFile:
                line = line.replace("SURFACE", "include/Upper_Surface")
                outFile.write(line)
    os.system("postProcess -latestTime -func boundaryCloud > log/boundaryCloud.log")
    step_Path = os.path.join(boundary_path, os.listdir(boundary_path)[0])
    os.rename(f"{step_Path}/surface_p.xy", f"{step_Path}/Upper.xy")

    # Post-process Lower Surface
    print("Post-processing Lower Surface.")
    with open("system/boundaryCloud_Template", "rt") as inFile:
        with open("system/boundaryCloud", "wt") as outFile:
            for line in inFile:
                line = line.replace("SURFACE", "include/Lower_Surface")
                outFile.write(line)
    os.system("postProcess -latestTime -func boundaryCloud > log/boundaryCloud.log")
    os.rename(f"{step_Path}/surface_p.xy", f"{step_Path}/Lower.xy")

    # Post-process internal cloud
    print("Post-processing internal cloud.")
    os.system("postProcess -latestTime -func internalCloud > log/internalCloud.log")

count = 1
for mesh in os.listdir(MESH_DIR):
    MESH_PATH = os.path.join(MESH_DIR, mesh)
    print(f"Processing mesh: {mesh}")

    os.system(f"cp {MESH_PATH} OpenFOAM/{mesh}")
    os.chdir("OpenFOAM/")
    os.system(f"fluent3DMeshToFoam {mesh} > log/MeshToFoam.log")
    os.chdir("..")

    for Mach in Mach_values:
        for angle in angle_values:
            fileName = f"{mesh}_{Mach}_{angle}"
            angle_radian = np.deg2rad(angle)
            Vel = V_sonic * Mach
            fsX = Vel * np.cos(angle_radian)
            fsZ = Vel * np.sin(angle_radian)

            print(f"Running simulation for case: {fileName}")
            os.chdir("OpenFOAM/")
            success = run_simulation(fsX, fsZ)

            if not success:
                os.chdir("..")
                continue  # Skip to the next Mach and angle if the simulation fails

            post_process(path=fileName)
            os.chdir("..")

            output = utils.outputProcessing(baseName=fileName, dataDir=outputDir, boundary_res=32, internal_res=128, Vel=V_sonic)
            output.output_npz()

            filelst = os.listdir("OpenFOAM")
            caseDict = int(np.max(get_numeric_files(filelst)))
            directory = f"CaseDict/{fileName}/"
            os.makedirs(directory, exist_ok=True)
            print(f"Copying case data to {directory}.")
            os.system(f"cp -r OpenFOAM/{caseDict}/ {directory}/{caseDict}")

            os.chdir("OpenFOAM/")
            os.system("./Allclean")
            os.chdir("..")

            print(f"Case {fileName} completed ({count}/{sample}).")
            count += 1

