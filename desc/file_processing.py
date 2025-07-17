import re
import os

def parse_job(file_path):
    """
    Parse the job input file to extract trajectory and sampling parameters.
    Returns a dictionary with keys: traj, frames, ref, atname, resname, qform.
    """
    variables = {
        "traj": None,
        "frames": None,
        "ref": None,
        "atname": None,
        "resname": None,
        "qform": None
    }
    patterns = {
        "traj":    re.compile(r'^traj\s*=\s*(\S+)'),
        "frames":  re.compile(r'^frames\s*=\s*(\d+)'),
        "ref":     re.compile(r'^ref\s*=\s*(\S+)'),
        "atname":  re.compile(r'^atname\s*=\s*(.+)$'),
        "resname": re.compile(r'^resname\s*=\s*(.+)$'),
        "qform":   re.compile(r'^qform\s*=\s*(.+)$')
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

    # Validate trajectory file extension
    if not variables["traj"].endswith('.pdb'):
        raise ValueError("Trajectory file must be a PDB file")

    # Validate frames as integer
    if variables["frames"] is not None and not variables["frames"].isdigit():
        raise ValueError("Frames must be an integer")

    # Ensure matching lengths for atom, residue names, and formulas
    if (variables["atname"] is not None and
        variables["resname"] is not None and
        variables["qform"] is not None):
        if not (len(variables["atname"]) == len(variables["resname"]) == len(variables["qform"])):
            raise ValueError("atname, resname, and qform must have the same number of elements")

    return variables


def cutdata(line, frame_number):
    """
    Parse a single PDB ATOM/HETATM line into structured data.
    Returns a list: [frame_number, atom_number, atom_name, residue_name, residue_number, x, y, z].
    """
    try:
        atom_number    = int(line[6:11])
        atom_name      = line[12:16].strip()
        residue_name   = line[17:20].strip()
        residue_number = int(line[22:26])
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        return [frame_number, atom_number, atom_name, residue_name, residue_number, x, y, z]
    except Exception as e:
        print(f"Error parsing line in frame {frame_number}: {line}")
        print(f"Exception: {e}")
        return None


def count_frames(input_file):
    """
    Determine the total number of frames by estimating frame size from the first two frames.
    Returns (total_frames, frame_size, box_length).
    """
    total_size = os.path.getsize(input_file)
    frame_size = 0
    end_count = 0

    with open(input_file, "r") as file:
        for line in file:
            if line.startswith("CRYST1"):
                # Extract box length from CRYST1 record
                box_length = float(line.split()[1])
            if line.strip() == "END":
                end_count += 1
                if end_count == 2:
                    break
            if end_count >= 1:
                frame_size += len(line)

    total_frames = int(total_size // frame_size)
    return total_frames, frame_size, box_length


def process_file(input_file, skip, total_frames, box_length, frame_size, current_frame):
    """
    Read only the frames needed based on skip factor, using efficient seeks for others.
    Accumulates data into column lists and returns:
      (frames, atom_nums, atom_names, residue_names, residue_nums,
       xs, ys, zs, last_frame_index, box_length)
    """
    if total_frames < skip:
        raise ValueError("Requested number of frames exceeds the total frames available.")

    sf = max(1, total_frames // skip)

    # Initialize column lists
    frames = []
    atom_nums = []
    atom_names = []
    residue_names = []
    residue_nums = []
    xs = []
    ys = []
    zs = []

    with open(input_file, "rb") as file:
        while current_frame[0] < total_frames:
            if current_frame[0] == 0:
                # First frame: read line by line until END
                while True:
                    raw = file.readline()
                    if not raw:
                        break
                    try:
                        line = raw.decode('ascii')
                    except UnicodeDecodeError:
                        line = raw.decode('latin-1')
                    line = line.rstrip()
                    if line == "END":
                        break
                    if line.startswith("ATOM  ") or line.startswith("HETATM"):
                        rec = cutdata(line, current_frame[0])
                        if rec:
                            frames.append(rec[0])
                            atom_nums.append(rec[1])
                            atom_names.append(rec[2])
                            residue_names.append(rec[3])
                            residue_nums.append(rec[4])
                            xs.append(rec[5])
                            ys.append(rec[6])
                            zs.append(rec[7])
            else:
                # Skip unneeded frames by seeking
                if current_frame[0] % sf != 0:
                    file.seek(frame_size, os.SEEK_CUR)
                else:
                    # Process this frame
                    chunk = file.read(frame_size)
                    for raw_line in chunk.split(b'\n'):
                        try:
                            line = raw_line.decode('ascii')
                        except UnicodeDecodeError:
                            line = raw_line.decode('latin-1')
                        line = line.rstrip()
                        if line == "END":
                            break
                        if line.startswith("ATOM  ") or line.startswith("HETATM"):
                            rec = cutdata(line, current_frame[0])
                            if rec:
                                frames.append(rec[0])
                                atom_nums.append(rec[1])
                                atom_names.append(rec[2])
                                residue_names.append(rec[3])
                                residue_nums.append(rec[4])
                                xs.append(rec[5])
                                ys.append(rec[6])
                                zs.append(rec[7])
            current_frame[0] += 1

    return (
        frames,
        atom_nums,
        atom_names,
        residue_names,
        residue_nums,
        xs,
        ys,
        zs,
        current_frame[0],
        box_length
    )
