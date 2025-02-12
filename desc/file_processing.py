import re
import os

def parse_job(file_path):
    variables = {
        "traj": None,
        "frames": None,
        "ref": None,
        "atname": None,
        "resname": None,
        "qform": None
        }
    
    patterns = {
        "traj": re.compile(r'^traj\s*=\s*(\S+)'),
        "frames": re.compile(r'^frames\s*=\s*(\d+)'),
        "ref": re.compile(r'^ref\s*=\s*(\S+)'),
        "atname": re.compile(r'^atname\s*=\s*(.+)$'),
        "resname": re.compile(r'^resname\s*=\s*(.+)$'),
        "qform": re.compile(r'^qform\s*=\s*(.+)$')
        }

    with open(file_path, 'r') as file:
        for line in file:
            for key, pattern in patterns.items():
                match = pattern.match(line)
                if match:
                    if key in ["atname", "resname", "qform"]:
                        variables[key] = match.group(1).split()
                    else:
                        variables[key] = match.group(1)

    # Validation checks
    if not variables["traj"].endswith('.pdb'):
        raise ValueError("Trajectory file must be a PDB file")

    if variables["frames"] is not None and not variables["frames"].isdigit():
        raise ValueError("Frames must be an integer")

    if variables["atname"] is not None and variables["resname"] is not None and variables["qform"] is not None:
        if not (len(variables["atname"]) == len(variables["resname"]) == len(variables["qform"])):
            raise ValueError("atname, resname, and qform must have the same number of elements")

    return variables

def cutdata(line, frame_number):
    try:
        # Apply fixed-width slicing based on the PDB column format
        # Columns:  frame_number, atom_number, atom_name, residue_name, residue_number, x, y, z
        atom_number = line[6:11].strip()        # Atom serial number
        atom_name = line[12:16].strip()         # Atom name
        residue_name = line[17:20].strip()      # Residue name
        residue_number = line[22:26].strip()    # Residue sequence number
        x = line[30:38].strip()                 # X coordinate
        y = line[38:46].strip()                 # Y coordinate
        z = line[46:54].strip()                 # Z coordinate
        return [frame_number, atom_number, atom_name, residue_name, residue_number, x, y, z]

    except Exception as e:
        print(f"Error parsing line in frame {frame_number}: {line}")
        print(f"Error: {e}")
        return None

def count_frames(input_file):
    total_size = os.path.getsize(input_file)
    frame_size = 0
    end_count = 0

    with open(input_file, "r") as file:
        for line in file:
            if line.startswith("CRYST1"):
                box_length = float(line.split()[1])  # Extract box length, only [1] if cubic
            if line.strip() == "END":
                end_count += 1
                if end_count == 2:
                    break
            if end_count >= 1:
                frame_size += len(line)

    total_frames = int(total_size // frame_size)
    return total_frames, frame_size, box_length

def process_file(input_file, skip, total_frames, box_length, frame_size, current_frame):
    frame_data = {}
    curfr_data = []

    if total_frames < skip:
        raise ValueError("Requested number of frames exceeds the total frames available.")

    sf = max(1, total_frames // skip)  # sf : Skip Factor (e.g., 1/10 = 10000 / 100)

    read_frames = 1 # Number of frames to read at once
    chunk_size = frame_size * read_frames

    with open(input_file, "r") as file:
        while current_frame[0] < total_frames:
            if current_frame[0] == 0:
                # Handle the first frame separately
                while True:
                    line = file.readline()
                    if not line:
                        break
                    line = line.rstrip()
                    prefix = line[:6]
                    if line == "END":
                        frame_data[current_frame[0]] = curfr_data[:]
                        curfr_data = []
                        break
                    elif prefix in {"ATOM  ", "HETATM"}:
                        filtered_line = cutdata(line, current_frame[0])
                        if filtered_line is not None:
                            curfr_data.append(filtered_line)
            else:
                chunk = file.read(chunk_size)
                lines = chunk.split('\n')
                if current_frame[0] % sf == 0:
                    for line in lines:
                        line = line.rstrip()
                        prefix = line[:6]
                        if line == "END":
                            frame_data[current_frame[0]] = curfr_data[:]
                            curfr_data = []
                            break
                        elif prefix in {"ATOM  ", "HETATM"}:
                            filtered_line = cutdata(line, current_frame[0])
                            if filtered_line is not None:
                                curfr_data.append(filtered_line)
            current_frame[0] += 1

    return frame_data, current_frame[0], box_length
