import os
import sys
import pandas as pd
from .file_processing import parse_job, process_file

def data_processing(job_file):
    print("  Reading Input File")

    ## Check if input is provided
    if len(sys.argv) != 2:
        print("No input file.")
        print("Usage: desc.py <job input>")
        sys.exit(1)

    ## Read and process the Job Input File
    job_variables = parse_job(job_file)

    # Extract individual variables
    traj = job_variables["traj"]
    frames = int(job_variables["frames"])
    ref = job_variables["ref"]
    atname = job_variables["atname"]
    resname = job_variables["resname"]
    qform = [int(q) for q in job_variables["qform"]]

    ## Process input file
    print("  Reading PDB Trajectory File")
    frame_data, total_frames, box_length = process_file(traj, frames)
    all_frame_data = [line for frame in frame_data for line in frame_data[frame]]

    maindf_datatypes = {
        "frame": "int32",
        "index": "int32",
        "atom": "category",
        "residue": "category",
        "residue_number": "int32",
        "x": "float32",
        "y": "float32",
        "z": "float32"
        }

    maindf = pd.DataFrame(all_frame_data, columns=list(maindf_datatypes.keys())).astype(maindf_datatypes)

    # Create and move into directory
    base_name = "DESC_JOB"
    job_id = 1

    while True:
        maindir = f"{base_name}{job_id}_{ref}"
        if not os.path.exists(maindir):
            os.makedirs(maindir)
            break
        job_id += 1

    os.chdir(maindir)

    ## MD Trajectory Data
    natoms_ss = maindf["index"].nunique()
    filt_ss = len(maindf) / natoms_ss

    return maindf, maindir, natoms_ss, filt_ss, box_length, ref, atname, resname, qform
