import os
import sys
import pandas as pd
import threading
import numpy as np
from .file_processing import parse_job, count_frames, process_file
from .utils import progress_bar

import time
import cProfile
import pstats

def data_processing(job_file):
    print("  Reading Input File")

    # Check if input is provided
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
    total_frames, frame_size, box_length = count_frames(traj)

    current_frame = [0]  # Mutable object to track current frame
    progress_thread = threading.Thread(target=progress_bar, args=(current_frame, total_frames, '  Processing File:', 'Complete', 1, 50, '█'))
    progress_thread.start()

    frame_data, current_frame_value, box_length = process_file(traj, frames, total_frames, box_length, frame_size, current_frame)

    all_frame_data = [line for frame in frame_data for line in frame_data[frame]]

    frame_data_array = np.array(all_frame_data, dtype=object)

    maindf = pd.DataFrame({
        "frame": frame_data_array[:, 0].astype(np.int32),
        "index": frame_data_array[:, 1].astype(np.int32) if np.issubdtype(frame_data_array[:, 1].dtype, np.number) else frame_data_array[:, 1].astype(str),
        "atom": pd.Categorical(frame_data_array[:, 2]),
        "residue": pd.Categorical(frame_data_array[:, 3]),
        "residue_number": frame_data_array[:, 4].astype(np.int32) if np.issubdtype(frame_data_array[:, 4].dtype, np.number) else frame_data_array[:, 4].astype(str),
        "x": frame_data_array[:, 5].astype(np.float32),
        "y": frame_data_array[:, 6].astype(np.float32),
        "z": frame_data_array[:, 7].astype(np.float32)
    })

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
   
    return maindf, maindir, box_length, ref, atname, resname, qform
