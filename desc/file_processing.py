import re

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
      
def process_file(input_file, skip):
    frame_data = {}
    current_frame = 0
    curfr_data = []
    box_length = None
    total_frames = 0
   
    with open(input_file, "r") as file:
        for line in file:
            if line.strip() == "END":
                total_frames += 1

    if total_frames < skip:
        raise ValueError("Requested number of frames exceeds the total frames available.")
    sf = max(1, total_frames // skip) # sf : Skip Factor (eg. 1/10 = 10000 / 100)

    with open(input_file, "r") as file:
        for line in file:
            if line.strip() == "END":
                if current_frame % sf == 0:
                    frame_data[current_frame] = curfr_data[:]
                curfr_data = []
                current_frame += 1
                
            elif line.startswith("CRYST1"):
                box_length = float(line.split()[1]) # Extract box length, only [1] if cubic

            elif line.startswith("ATOM") or line.startswith("HETATM"):
                if current_frame % sf == 0:
                    filtered_line = cutdata(line, current_frame)
                    curfr_data.append(filtered_line)

        return frame_data, current_frame, box_length                

def cutdata(line, frame_number):
    parts = line.split()
    return [frame_number] + [parts[i] for i in [1, 2, 3, 5, 6, 7, 8]]