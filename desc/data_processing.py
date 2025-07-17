import os
import sys
import pandas as pd
import threading
import time

from .file_processing import parse_job, count_frames, process_file
from .utils import progress_bar

def data_processing(job_file):
    """
    Main pipeline: read parameters, profile I/O, build DataFrame, and set up output.
    """
    print("  Reading Input File")

    # Validate input argument count
    if len(sys.argv) != 2:
        print("No input file.")
        print("Usage: desc.py <job input>")
        sys.exit(1)

    # Parse job parameters
    job_vars = parse_job(job_file)
    traj = job_vars["traj"]
    frames_to_sample = int(job_vars["frames"])
    ref = job_vars["ref"]
    atname = job_vars["atname"]
    resname = job_vars["resname"]
    qform = job_vars["qform"]

    # Estimate total frames, frame size, and box length
    total_frames, frame_size, box_length = count_frames(traj)

    # Launch progress bar in a background thread
    current_frame = [0]
    progress_thread = threading.Thread(
        target=progress_bar,
        args=(
            current_frame,
            total_frames,
            "  Processing File:",
            "Complete",
            1,    # decimals
            50,   # bar length
            "█"   # fill character
        )
    )
    progress_thread.start()

    # Process only the needed frames; returns 10 values
    frames, atom_nums, atom_names, residue_names, residue_nums, xs, ys, zs, last_frame, box_length = \
        process_file(
            traj,
            frames_to_sample,
            total_frames,
            box_length,
            frame_size,
            current_frame
        )

    # Wait until the progress bar finishes
    progress_thread.join()

    # Build the DataFrame directly from column lists
    maindf = pd.DataFrame({
        "frame":          frames,
        "index":          atom_nums,
        "atom":           pd.Categorical(atom_names),
        "residue":        pd.Categorical(residue_names),
        "residue_number": residue_nums,
        "x":               xs,
        "y":               ys,
        "z":               zs
    })

    # Create a new output directory and switch into it
    base_name = "DESC_JOB"
    job_id = 1
    while True:
        maindir = f"{base_name}{job_id}_{ref}"
        if not os.path.exists(maindir):
            os.makedirs(maindir)
            break
        job_id += 1
    os.chdir(maindir)

    # Compute some summary statistics
    natoms_ss = maindf["index"].nunique()
    filt_ss = len(maindf) / natoms_ss

    return maindf, maindir, box_length, ref, atname, resname, qform
