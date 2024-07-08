import sys
import os
import time

from .file_processing import parse_job, process_file
from .file_generation import ADF_input
from .geometry import ref_radius, com_calc, compute_vRg_frame, avg_vRg, calculate_rdf
from .constants import atomic_weights, vol_dict, adf_solvent
from .utils import getelement, isbtw
from .data_processing import data_processing
from .solvent_analysis import analyze_solvent
from .ref_analysis import analyze_reference
from .residue_analysis import process_residues
from .dependency_installer import install_and_import

def main():
    start_time = time.time()

    if len(sys.argv) != 2:
        print("No input file.")
        print("Usage: desc.py <job input>")
        sys.exit(1)

    job_file = sys.argv[1]
    maindf, maindir, natoms_ss, filt_ss, box_length, ref, atname, resname, qform = data_processing(job_file)
  
    ## Solvent
    solv_name, eps = analyze_solvent(maindf, adf_solvent)

    ## Reference
    ref_com, r_ref, adf_atomlist = analyze_reference(maindf, ref, atomic_weights)

    ## Residues          
    process_residues(maindf, ref_com, box_length, vol_dict, atomic_weights, eps, ref, r_ref, adf_atomlist, solv_name, atname, resname, qform)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"  Execution time: {int(elapsed_time)} seconds")

    # Add execution time to log files
    add_execution_time_to_logs(atname, resname, int(elapsed_time))

def add_execution_time_to_logs(atname, resname, execution_time):
    if isinstance(atname, str):
        atnames = atname.split()
    else:
        atnames = atname

    if isinstance(resname, str):
        resnames = resname.split()
    else:
        resnames = resname

    for ATNAME, RESNAME in zip(atnames, resnames):
        directory = f"{ATNAME}_{RESNAME}"
        log_filename = os.path.join(directory, f"logfile_{ATNAME}_{RESNAME}")
        if os.path.exists(log_filename):
            with open(log_filename, 'a') as f:
                f.write(f"Execution time: {execution_time} seconds\n")
                
if __name__ == "__main__":
    dependencies = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'sklearn_extra': 'sklearn_extra',
        'openbabel': 'openbabel',
        'rdkit': 'rdkit'
    }
    for package, import_name in dependencies.items():
        install_and_import(package, import_name)

    main()