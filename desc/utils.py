import os
import sys
import time

def getelement(atom_str):
    if len(atom_str) > 1 and atom_str[1].isalpha():
        return atom_str[0] + atom_str[1].lower()
    else:
        return atom_str[0]

def isbtw(n):
    decimal_part = n % 1
    near_aps = far_aps = None
    chkaps = False
    if 0.22 <= decimal_part < 0.5:
        near_aps = int(n)
        far_aps = int(n) + 1
        chkaps = True
    elif 0.5 <= decimal_part <= 0.78:
        near_aps = int(n) + 1
        far_aps = near_aps - 1
        chkaps = True
    return (near_aps, far_aps), chkaps

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

def howtocite():
    cite_message = """
    +------------------------------------------------------------+
    |                                                            |
    |            Thank you for using DESC!                       |
    |                                                            |
    | Your contribution and feedback are invaluable to us.       |
    |                                                            |
    |                                                            |
    | If you use DESC in your research or project, please cite   |
    | it as follows:                                             |
    |                                                            |
    | Albert Masip-Sánchez, Josep M. Poblet, Xavier López. 2024  |
    | DESC: Dynamic Environment in Solution by Clustering.       |
    | Version 1.0.0. Retriveved from                             |
    | https://github.com/qcgurv/DESC                             |
    |                                                            |
    +------------------------------------------------------------+
    """
    print(cite_message)

def progress_bar(current_frame, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Interactive progress bar function
    @params:
        current_frame - Required : list containing current iteration (List[int])
        total         - Required : total iterations (Int)
        prefix        - Optional : prefix string (Str)
        suffix        - Optional : suffix string (Str)
        decimals      - Optional : positive number of decimals in percent complete (Int)
        length        - Optional : character length of bar (Int)
        fill          - Optional : bar fill character (Str)
    """
    while current_frame[0] < total:
        percent = ("{0:." + str(decimals) + "f}").format(100 * (current_frame[0] / float(total)))
        filled_length = int(length * current_frame[0] // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
        sys.stdout.flush()
        time.sleep(0.1)  # Update frequently to reflect incremental progress

    # Print the final bar once done
    percent = ("{0:." + str(decimals) + "f}").format(100 * (total / float(total)))
    filled_length = int(length * total // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}\n')
    sys.stdout.flush()
