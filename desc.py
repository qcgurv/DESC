#!/bin/env python3.9

import os
import re
import sys
import numpy as np
import pandas as pd
from sklearn_extra.cluster import KMedoids
from scipy.spatial.distance import pdist, squareform
from scipy.signal import find_peaks, savgol_filter

## Atomic dictionaries
atomic_weights = {
    'H' : 1.008,'He' : 4.003, 'Li' : 6.941, 'Be' : 9.012,
    'B' : 10.811, 'C' : 12.011, 'N' : 14.007, 'O' : 15.999,
    'F' : 18.998, 'Ne' : 20.180, 'Na' : 22.990, 'Mg' : 24.305,
    'Al' : 26.982, 'Si' : 28.086, 'P' : 30.974, 'S' : 32.066,
    'Cl' : 35.453, 'Ar' : 39.948, 'K' : 39.098, 'Ca' : 40.078,
    'Sc' : 44.956, 'Ti' : 47.867, 'V' : 50.942, 'Cr' : 51.996,
    'Mn' : 54.938, 'Fe' : 55.845, 'Co' : 58.933, 'Ni' : 58.693,
    'Cu' : 63.546, 'Zn' : 65.38, 'Ga' : 69.723, 'Ge' : 72.631,
    'As' : 74.922, 'Se' : 78.971, 'Br' : 79.904, 'Kr' : 84.798,
    'Rb' : 84.468, 'Sr' : 87.62, 'Y' : 88.906, 'Zr' : 91.224,
    'Nb' : 92.906, 'Mo' : 95.95, 'Tc' : 98.907, 'Ru' : 101.07,
    'Rh' : 102.906, 'Pd' : 106.42, 'Ag' : 107.868, 'Cd' : 112.414,
    'In' : 114.818, 'Sn' : 118.711, 'Sb' : 121.760, 'Te' : 126.7,
    'I' : 126.904, 'Xe' : 131.294, 'Cs' : 132.905, 'Ba' : 137.328,
    'La' : 138.905, 'Ce' : 140.116, 'Pr' : 140.908, 'Nd' : 144.243,
    'Pm' : 144.913, 'Sm' : 150.36, 'Eu' : 151.964, 'Gd' : 157.25,
    'Tb' : 158.925, 'Dy': 162.500, 'Ho' : 164.930, 'Er' : 167.259,
    'Tm' : 168.934, 'Yb' : 173.055, 'Lu' : 174.967, 'Hf' : 178.49,
    'Ta' : 180.948, 'W' : 183.84, 'Re' : 186.207, 'Os' : 190.23,
    'Ir' : 192.217, 'Pt' : 195.085, 'Au' : 196.967, 'Hg' : 200.592,
    'Tl' : 204.383, 'Pb' : 207.2, 'Bi' : 208.980, 'Po' : 208.982,
    'At' : 209.987, 'Rn' : 222.081, 'Fr' : 223.020, 'Ra' : 226.025,
    'Ac' : 227.028, 'Th' : 232.038, 'Pa' : 231.036, 'U' : 238.029,
    'Np' : 237, 'Pu' : 244, 'Am' : 243, 'Cm' : 247
}

vol_dict = {
    "H": 1.350, "He": 1.275, "Li": 2.125, "Be": 1.858,
    "B": 1.792, "C": 1.700, "N": 1.608, "O": 1.517,
    "F": 1.425, "Ne": 1.333, "Na": 2.250, "Mg": 2.025,
    "Al": 1.967, "Si": 1.908, "P": 1.850, "S": 1.792,
    "Cl": 1.725, "Ar": 1.658, "K": 2.575, "Ca": 2.342,
    "Sc": 2.175, "Ti": 1.992, "V": 1.908, "Cr": 1.875,
    "Mn": 1.867, "Fe": 1.858, "Co": 1.858, "Ni": 1.850,
    "Cu": 1.883, "Zn": 1.908, "Ga": 2.050, "Ge": 2.033,
    "As": 1.967, "Se": 1.908, "Br": 1.850, "Kr": 1.792,
    "Rb": 2.708, "Sr": 2.500, "Y": 2.258, "Zr": 2.117,
    "Nb": 2.025, "Mo": 1.992, "Tc": 1.967, "Ru": 1.950,
    "Rh": 1.950, "Pd": 1.975, "Ag": 2.025, "Cd": 2.083,
    "In": 2.200, "Sn": 2.158, "Sb": 2.100, "Te": 2.033,
    "I": 1.967, "Xe": 1.900, "Cs": 2.867, "Ba": 2.558,
    "La": 2.317, "Ce": 2.283, "Pr": 2.275, "Nd": 2.275,
    "Pm": 2.267, "Sm": 2.258, "Eu": 2.450, "Gd": 2.258,
    "Tb": 2.250, "Dy": 2.242, "Ho": 2.225, "Er": 2.225,
    "Tm": 2.225, "Yb": 2.325, "Lu": 2.208, "Hf": 2.108,
    "Ta": 2.025, "W": 1.992, "Re": 1.975, "Os": 1.958,
    "Ir": 1.967, "Pt": 1.992, "Au": 2.025, "Hg": 2.108,
    "Tl": 2.158, "Pb": 2.283, "Bi": 2.217, "Po": 2.158,
    "At": 2.092, "Rn": 2.025, "Fr": 3.033, "Ra": 2.725,
    "Ac": 2.567, "Th": 2.283, "Pa": 2.200, "U": 2.100,
    "Np": 2.100, "Pu": 2.100
}

adf_solvent = {
    'AceticAcid': 'C2H4O2',
    'Acetone': 'C3H6O',
    'Acetonitrile': 'C2H3N',
    'Ammonia': 'H3N',
    'Aniline': 'C6H7N',
    'Benzene': 'C6H6',
    'BenzylAlcohol': 'C7H8O',
    'Bromoform': 'CHBr3',
    'Butanol': 'C4H10O',
    'isoButanol': 'C4H10O',
    'tertButanol': 'C4H10O',
    'CarbonDisulfide': 'CS2',
    'CarbonTetrachloride': 'CCl4',
    'Chloroform': 'CHCl3',
    'Cyclohexane': 'C6H12',
    'Cyclohexanone': 'C6H10O',
    'Dichlorobenzene': 'C6H4Cl2',
    'DiethylEther': 'C4H10O',
    'Dioxane': 'C4H8O2',
    'DMFA': 'C3H7NO',
    'DMSO': 'C2H6OS',
    'Ethanol': 'C2H6O',
    'EthylAcetate': 'C4H8O2',
    'Dichloroethane': 'C2H4Cl2',
    'EthyleneGlycol': 'C2H6O2',
    'Formamide': 'CH3NO',
    'FormicAcid': 'CH2O2',
    'Glycerol': 'C3H8O3',
    'HexamethylPhosphoramide': 'C6H18N3OP',
    'Hexane': 'C6H14',
    'Hydrazine': 'H4N2',
    'Methanol': 'CH4O',
    'MethylEthylKetone': 'C4H8O',
    'Dichloromethane': 'CH2Cl2',
    'Methylformamide': 'C2H5NO',
    'Methypyrrolidinone': 'C5H9NO',
    'Nitrobenzene': 'C6H5NO2',
    'Nitrogen': 'N2',
    'Nitromethane': 'CH3NO2',
    'PhosphorylChloride': 'Cl3OP',
    'IsoPropanol': 'C3H8O',
    'Pyridine': 'C5H5N',
    'Sulfolane': 'C4H8O2S',
    'Tetrahydrofuran': 'C4H8O',
    'Toluene': 'C7H8',
    'Triethylamine': 'C6H15N',
    'TrifluoroaceticAcid': 'C2HF3O2',
    'Water': 'H2O'
}

## DEFINE FUNCTIONS

def parse_job(file_path):
    variables = {
        "traj": None,
        "frames": None,
        "ref": None,
        "eps": None,
        "atname": None,
        "resname": None,
        "qform": None
        }
    
    patterns = {
        "traj": re.compile(r'^traj\s*=\s*(\S+)'),
        "frames": re.compile(r'^frames\s*=\s*(\d+)'),
        "ref": re.compile(r'^ref\s*=\s*(\S+)'),
        "eps": re.compile(r'^eps\s*=\s*(\S+)'),
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

    if variables["eps"] is not None:
        try:
            float(variables["eps"])
        except ValueError:
            raise ValueError("Dielectric constant must be a number")

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

def getelement(atom_str):
    return atom_str[:2] if len(atom_str) > 1 and atom_str[1].isalpha() else atom_str[0]

def ref_radius(coordinates):
    r_ref = 0
    max_dist = 0
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            dist = np.linalg.norm(coordinates[i] - coordinates[j])
            max_dist = max(max_dist, dist)
            r_ref = max_dist / 2
    return r_ref
            
def isbtw(n):
    decimal_part = n % 1
    near_aps = far_aps = None
    chkaps = False
    if 0.31 <= decimal_part < 0.5:
        near_aps = int(n)
        far_aps = int(n) + 1
        chkaps = True
    elif 0.5 <= decimal_part <= 0.69:
        near_aps = int(n) + 1
        far_aps = near_aps - 1
        chkaps = True
    return (near_aps, far_aps), chkaps

def com_calc(data, atomic_weights):
    elements = np.array([getelement(atom) for atom in data[:, 0]])
    coordinates = data[:, 1:4].astype(np.float32)
    atomic_weights_vector = np.array([atomic_weights.get(elem, 1.0) for elem in elements], dtype=np.float32)
    weighted_coordinates = coordinates * atomic_weights_vector[:, np.newaxis]
    total_mass = atomic_weights_vector.sum()
    com = weighted_coordinates.sum(axis=0) / total_mass
    return com

def radius_of_gyration(atoms, vol_dict):
    atom_data = np.array([[atom['type'], *atom['coords']] for atom in atoms])
    center_of_mass = com_calc(atom_data, vol_dict)
    atomic_weights_vector = np.array([vol_dict[getelement(atom)] for atom in atom_data[:, 0]], dtype=np.float32)
    coordinates = atom_data[:, 1:4].astype(np.float32)
    sum_squared = np.sum(atomic_weights_vector * np.linalg.norm(coordinates - center_of_mass, axis=1)**2)
    total_mass = atomic_weights_vector.sum()
    return np.sqrt(sum_squared / total_mass)

def compute_vRg_frame(df, vol_dict):
    results = []
    grouped = df.groupby(['frame', 'residue_number'])

    for (frame, residue_number), group in grouped:
        atoms = [{'type': atom, 'coords': (x, y, z)} for atom, x, y, z in zip(group['atom'], group['x'], group['y'], group['z'])]
        rg = radius_of_gyration(atoms, vol_dict)
        results.append({'residue_number': residue_number, 'v-Rg': rg})
    return pd.DataFrame(results)

def avg_vRg(df):
    return df.groupby('residue_number')['v-Rg'].mean().reset_index()

def calculate_rdf(df, com, box_length, bin_width=0.1):
    half_box_length = box_length / 2
    volume = box_length**3
    number_density = len(df) / volume

    def pbc_distance(positions, ref_point, box_length):
        # Apply minimum image convention
        delta = positions - ref_point
        delta = delta - np.rint(delta / box_length) * box_length
        return np.sqrt((delta**2).sum(axis=1))

    distances = pbc_distance(df[['x', 'y', 'z']].values, com, box_length)

    r_max = half_box_length
    num_bins = int(r_max / bin_width)
    bin_edges = np.linspace(0, r_max, num_bins + 1)
    bin_r = (bin_edges[:-1] + bin_edges[1:]) / 2
    rdf_counts = np.zeros(num_bins)

    # Calculate RDF
    for d in distances:
        bin_index = int(d / bin_width)
        if bin_index < num_bins:
            rdf_counts[bin_index] += 1

    # Normalize RDF
    rdf = rdf_counts / (4/3 * np.pi * ((np.arange(1, num_bins + 1) * bin_width)**3 - (np.arange(0, num_bins) * bin_width)**3))
    rdf /= number_density

    # Save RDF data to a text file
    rdf_data = np.column_stack((bin_r, rdf))
    np.savetxt('rdf.dat', rdf_data, fmt='%0.6f')
    
    return bin_r, rdf, bin_width, distances, number_density, bin_edges

def ADF_input(ref, atomlist, solv_name, ghost, pointcharges, val=None):
    if val != 0:
        filename = f"{ref}_DESC_{val}.in"
    else:
        filename = f"{ref}_DESC.in"

    input_content = f"""## Default theory level: BP86 / TZP
## The solvent has been automatically detected. Please ensure it is correct.
## Please enter the charge (and spin polarization, if necessary).

Task SinglePoint

Engine ADF
    Basis
        Type TZP
        core small
        createoutput no
    End
    XC
        GGA BP86
    End
    Title DESC input file
    Solvation
        Surf Delley
        Solv name={solv_name} cav0=0.0 cav1=0.0067639
        Charged method=CONJ
        C-Mat POT
        SCF VAR ALL
        CSMRSP
    End
    NumericalQuality Good
EndEngine

System
    Atoms
{atomlist}{ghost}    End
    Charge Q
    ElectrostaticEmbedding
        MultipolePotential
            Coordinates
{pointcharges}            End
        End
    End
End
"""
    with open(filename, "w") as file:
        file.write(input_content)

print("  Reading Input File")

## Check if input is provided
if len(sys.argv) != 2:
    print("No input file.")
    print("Usage: desc.py <job input>")
    sys.exit(1)

## Read and process the Job Input File
job_variables = parse_job(sys.argv[1])

# Extract individual variables
traj = job_variables["traj"]
frames = int(job_variables["frames"])
ref = job_variables["ref"]
eps = float(job_variables["eps"])
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

## Solvent
ff = maindf[maindf['frame'] == maindf['frame'].min()]   # ff: First Frame, only to identify the solvent resname
solv_resname = ff['residue'].value_counts().idxmax()
solvmatch = ff[ff['residue'] == solv_resname].iloc[0]
residue_number = solvmatch['residue_number']

filt_ff = ff[(ff['residue_number']  == residue_number) & (ff['residue'] == solv_resname)]
atom_counts = {}
for atom in filt_ff['atom']:
    if atom[-1].isdigit():
        element = atom[:-1] # Remove the digit at the end
    else:
        element = atom
    if len(element) > 1:
        element = element[0] + element[1:].lower() # Make second letter lowercase

    if element in atom_counts:
        atom_counts[element] += 1
    else:
        atom_counts[element] = 1

solv_formula = ''.join(f"{element}{count if count > 1 else ''}" for element, count in sorted(atom_counts.items()))

# Standardize the solv_formula
solv_formula_expanded = re.sub(r'\(([^)]+)\)(\d*)', lambda m: m.group(1) * int(m.group(2) or 1), solv_formula)
elements = re.findall(r'([A-Z][a-z]?)(\d*)', solv_formula_expanded)
element_counts = {}
for element, count in elements:
    element_counts[element] = element_counts.get(element, 0) + int(count or 1)
standardized_solv_formula = ''.join(f"{element}{count if count > 1 else ''}" for element, count in sorted(element_counts.items()))

# Compare the standardized formula with the dictionary
for name, chem_formula in adf_solvent.items():
    # Standardize the dictionary formula
    chem_formula_expanded = re.sub(r'\(([^)]+)\)(\d*)', lambda m: m.group(1) * int(m.group(2) or 1), chem_formula)
    elements = re.findall(r'([A-Z][a-z]?)(\d*)', chem_formula_expanded)
    element_counts = {}
    for element, count in elements:
        element_counts[element] = element_counts.get(element, 0) + int(count or 1)
    standardized_chem_formula = ''.join(f"{element}{count if count > 1 else ''}" for element, count in sorted(element_counts.items()))

    if standardized_chem_formula == standardized_solv_formula:
        solv_name = name
        break
else:
    solv_name = "Formula not found"

## Reference data
print(f"  Looking for {ref}")
ref_data = maindf[maindf['residue'] == ref]
natoms_ref = ref_data['index'].nunique()

## COM of REF
ref_group = ref_data.groupby(['frame'])
ref_coms = []
ref_radii = []

for frame, group in ref_group:
    ref_com_frame = com_calc(group[['atom','x','y','z']].to_numpy(), atomic_weights)
    ref_coms.append(ref_com_frame)

    ref_coord = group[['x','y','z']].to_numpy()
    r_ref_frame = ref_radius(ref_coord)
    ref_radii.append(r_ref_frame)

ref_com = np.mean(ref_coms, axis=0) # Average Center of Mass of REF
r_ref = np.mean(ref_radii)  # Average Radius of REF

## Average structure of REF
atomdata = ref_data[['index','atom']]
xyzdata = ref_data[['index','x','y','z']]
avgxyz = xyzdata.groupby('index').mean().reset_index()

avgref = pd.merge(atomdata.drop_duplicates(), avgxyz, on='index')

xyz_ref = f"{ref}.xyz"
with open(xyz_ref, "w") as xyzfile:
    xyzfile.write(f"{natoms_ref}\n")
    xyzfile.write(f"Average coordinates of {ref} residue\n")
    
    for _, row in avgref.iterrows():
        elemref = getelement(row['atom'])
        x, y, z, = float(row['x']), float(row['y']), float(row['z'])
        xyzfile.write(f"{elemref:<2s} {x:>8.3f} {y:>8.3f} {z:>8.3f}\n")
print(f"  Average structure written to {xyz_ref}")

# Store xyz_ref in adf_atomlist for ADF input
with open(xyz_ref, 'r') as file:
    xyz_lines = file.readlines()[2:]

adf_atomlist = ''.join(xyz_lines)

## Residues          
atnames = job_variables['atname']
resnames = job_variables['resname']
qforms = [int(q) for q in job_variables['qform']]

for ATNAME, RESNAME, CHARGE in zip(atnames, resname, qforms):
    print(f"  Looking for {ATNAME}-{RESNAME}")
    
    # Create and move into directory
    directory = f"{ATNAME}_{RESNAME}"
    os.mkdir(directory)
    os.chdir(directory)

    # Residue data
    resid_data = maindf[(maindf['residue'] == RESNAME)]
    res_data = maindf[(maindf['atom'] == ATNAME) & (maindf['residue'] == RESNAME)]
    natoms_res = res_data['residue_number'].nunique()
    
    # Volume-weighted Radius of Gyration
    #print(resid_data)
    vRg_df = compute_vRg_frame(resid_data, vol_dict)
    avg_vRg_df = avg_vRg(vRg_df)
    mean_vRg = avg_vRg_df['v-Rg'].mean()
    std_vRg = avg_vRg_df['v-Rg'].std()
    print(f"  v-Rg of {RESNAME}: {mean_vRg:.3f} ± {std_vRg:.3f}")

    # Radial Distribution Function between COM(REF) and ATNAME
    bin_r, g_r, bin_width, distances, number_density, bin_edges = calculate_rdf(res_data, ref_com, box_length, bin_width=0.1)
    rdfdf = pd.DataFrame({"Distance": bin_r, "g(r)": g_r})
    rdf_name = 'rdf_' + ref + '_' + ATNAME + '_' + RESNAME + '.dat'
    os.rename('rdf.dat', rdf_name)

    # Apply Savitzky-Golay filter to smooth the RDF
    g_r_smoothed = savgol_filter(g_r, window_length=11, polyorder=2)

    # Compute the first derivative of the smoothed RDF
    g_r_fd = savgol_filter(g_r, window_length=11, polyorder=2, deriv=1)

    # Find peaks
    h = 0.5 * g_r_smoothed.max()
    dist = 10 * bin_width
    prom = np.std(g_r_smoothed)

    peaks, _ = find_peaks(g_r_smoothed, height=h, distance=dist, prominence=prom)

    res_peak_max_r = []
    res_integration_distances = []

    # Identify each peak maxima and integration distance
    for peak_index in peaks:
        res_peak_max_r.append(bin_r[peak_index])
        
        # Find where the first derivative changes sign from negative to positive
        for i in range(peak_index + 1, len(g_r_fd) - 1):
         if g_r_fd[i - 1] < 0 and g_r_fd[i] >= 0:
             res_integration_distances.append(bin_r[i])
             break
        else:
            # If no sign change
            res_integration_distances.append(None)

    # Filter COM(REF)-ATNAME(RESNAME) by Integration Distance
    res_data = res_data.copy()
    res_data['dist'] = distances
    DIST = np.max(res_integration_distances)
    intres_data = res_data[res_data['dist'] <= DIST]
    HPD = np.max(res_peak_max_r) # Distance of the highest peak of RDF (often the 1st peak)
    print(f"  Highest Peak Distance: {HPD:.2f} angstrom")
    print(f"  Total Integration Distance: {DIST:.2f} angstrom")

    # Pointcharges
    filt_at = len(intres_data)
    int_ss = int(len(res_data) / natoms_res)
    aps = filt_at / int_ss # aps = Atoms Per Snapshot
    print(f"  Clustering...")

    print_once = False
    
    (near_aps, far_aps), chkaps = isbtw(aps)
    if chkaps:
        for val in  [near_aps, far_aps]:
            if not print_once:
                if val == 0:
                    continue
                near = intres_data[['x','y','z']].to_numpy()
                kmed = KMedoids(n_clusters=val, method='alternate', init='heuristic', random_state=42, max_iter=10000).fit(near)
                klabels = kmed.labels_
                intres_data = intres_data.copy()
                intres_data['Cluster'] = klabels
                cluster = kmed.cluster_centers_
                # End of cluster analysis.
            
                # Random selection of pointcharges
                unique_frames = intres_data['frame'].unique()
                pqframes = 100 # Number of frames to select
                if len(unique_frames) >= pqframes:   # Ensure that there are at least "pqframes" unique frames
                    sel_frames =pd.Series(unique_frames).sample(n=pqframes)
                else:
                    raise ValueError("Not enough unique frames to sample from.")
                
                pqdf_sel = intres_data[intres_data['frame'].isin(sel_frames)]
                pqdf_sel = pqdf_sel.copy()
                
                # Pointcharges calculation
                N_tot = len(pqdf_sel)
                pqdf_sel['q_fitt'] = (CHARGE * val / N_tot) * (0.025 + 1 / eps)
                pqres_data = pqdf_sel[['x','y','z','q_fitt']]

                # Save cluster
                clusterdf = pd.DataFrame(cluster, columns = ['x','y','z'])
                for _, row in clusterdf.iterrows():
                    pqres_data = pqres_data[~((pqres_data['x'] == row['x']) &
                                            (pqres_data['y'] == row['y']) &
                                            (pqres_data['z'] == row['z']))] 
                ghostdf = clusterdf.copy()
                ghostdf.insert(0, 'atom', 'Gh.H')
                ghostdf['adf.r'] = mean_vRg
                clusterdf.insert(0, 'atom', 'He')

                clustname = f"ghostatoms_{val}.xyz"
                with open(clustname, 'w') as f:
                    f.write(f"{len(clusterdf)}\n")
                    f.write("Ghost atoms XYZ. Helium atoms were placed instead of Gh.H to be able to visualize them.\n")
                    for _, row in clusterdf.iterrows():
                        f.write(f"{row['atom']:<5} {row['x']:>10.5f} {row['y']:>10.5f} {row['z']:>10.5f}\n")

                print(f"  There are {aps:.2f} {ATNAME}-{RESNAME} atoms in average coordinating the solute.")
                print_once = True

                adf_ghost = ""
                for _, row in ghostdf.iterrows():
                    adf_ghost += f"{row['atom']:<2s} {row['x']:>8.3f} {row['y']:>8.3f} {row['z']:>8.3f} adf.r={row['adf.r']:>5.3f}\n"

                # Check minimum distance between two points. Minimum distance is 0.0165 angstroms (in ADF).
                print(f"  N = {val}. Checking if there are atoms too close...")
                pqres_data = pqres_data[['x','y','z','q_fitt']].copy()
                distances = pdist(pqres_data[['x','y','z']].values)
                dist_matrix = squareform(distances)

                pq_data = pqres_data.copy()

                pqres_data.reset_index(drop=True, inplace=True)
                pq_data.reset_index(drop=True, inplace=True)

                # Find points close to each other
                droppoints = set()

                for i in range(dist_matrix.shape[0]):
                    if i not in droppoints:  # Check if the point hasn't already been flagged for deletion
                        for j in range(i+1, dist_matrix.shape[1]):
                            if dist_matrix[i, j] <= 0.0165:
                                # Add the charge of j to i
                                pq_data.at[i, 'q_fitt'] += pqres_data.at[j, 'q_fitt']
                                # Flag j for deletion
                                droppoints.add(j)

                # Drop rows that have been flagged
                pq_data = pq_data.drop(index=droppoints)
                pq_data.reset_index(drop=True, inplace=True)

                numdrop = len(pqres_data) - len(pq_data)

                if numdrop != 0:
                    print(f"  N = {val}. {numdrop} point(s) were substracted from the point charges for being too close to the others.")
                else:
                    print(f"  N = {val}. No points are too close to the others.")

                pointchargesdf = pq_data.copy()
                pq_data = pq_data.drop(columns=['q_fitt'])
                pq_data.insert(0, 'atom', 'Ar')
                
                pqname = f"pointcharges_{val}.xyz"
                with open(pqname, 'w') as f:
                    f.write(f"  {len(pq_data)}\n")
                    f.write(f"   Point charges XYZ. Argon atoms were placed in order to visualize them, and q_fitt was drop it. q_fitt = {pqdf_sel['q_fitt'].iloc[0]:>7.6f}.\n")
                    for _, row in pq_data.iterrows():
                        f.write(f"{row['atom']:<5} {row['x']:>10.5f} {row['y']:>10.5f} {row['z']:>10.5f}\n")

                adf_pointcharges = ""
                for _, row in pointchargesdf.iterrows():
                    adf_pointcharges += f"{row['x']:>8.3f} {row['y']:>8.3f} {row['z']:>8.3f} {row['q_fitt']:>8.6f}\n"
                
                ADF_input(ref, adf_atomlist, solv_name, adf_ghost, adf_pointcharges, val)
                os.rename(f"./{ref}_DESC_{val}.in", f"../{ref}_DESC_{val}.in")

    else:
        if round(aps) == 0:
            print(" -- DONE!")
            print(f"  NOTE: THERE ARE NO {RESNAME} NEAR {ref}. NO CLUSTER ANALYSIS HAS BEEN PERFORMED.")
            break       
        else:     
            r_aps = round(aps)
            near = intres_data[['x','y','z']].to_numpy()
            kmed = KMedoids(n_clusters=r_aps, method='alternate', init='heuristic', random_state=42, max_iter=10000).fit(near)
            klabels = kmed.labels_
            intres_data = intres_data.copy()
            intres_data['Cluster'] = klabels
            cluster = kmed.cluster_centers_
            # End of cluster analysis.

            # Random selection of pointcharges
            unique_frames = intres_data['frame'].unique()
            pqframes = 100 # Number of frames to select
            if len(unique_frames) >= pqframes:   # Ensure that there are at least "pqframes" unique frames
                sel_frames =pd.Series(unique_frames).sample(n=pqframes)
            else:
                raise ValueError("Not enough unique frames to sample from.")
            
            pqdf_sel = intres_data[intres_data['frame'].isin(sel_frames)]
            pqdf_sel = pqdf_sel.copy()

            # Pointcharges calculation
            N_tot = len(pqdf_sel)
            pqdf_sel['q_fitt'] = (CHARGE * r_aps / N_tot) * (0.025 + 1 / eps)
            pqres_data = pqdf_sel[['x','y','z','q_fitt']]

            # Save cluster
            clusterdf = pd.DataFrame(cluster, columns = ['x','y','z'])
            for _, row in clusterdf.iterrows():
                pqres_data = pqres_data[~((pqres_data['x'] == row['x']) &
                                          (pqres_data['y'] == row['y']) &
                                          (pqres_data['z'] == row['z']))] 
            ghostdf = clusterdf.copy()
            ghostdf.insert(0, 'atom', 'Gh.H')
            ghostdf['adf.r'] = mean_vRg
            clusterdf.insert(0, 'atom', 'He')

            with open('ghostatoms.xyz', 'w') as f:
                f.write(f"  {len(clusterdf)}\n")
                f.write("   Ghost atoms XYZ. Helium atoms were placed instead of Gh.H to be able to visualize them.\n")
                for _, row in clusterdf.iterrows():
                    f.write(f"{row['atom']:<5} {row['x']:>10.5f} {row['y']:>10.5f} {row['z']:>10.5f}\n")

            print(f"  There are {aps:.2f} {ATNAME}-{RESNAME} atoms in average coordinating the solute.")

            adf_ghost = ""
            for _, row in ghostdf.iterrows():
                adf_ghost += f"{row['atom']:<2s} {row['x']:>8.3f} {row['y']:>8.3f} {row['z']:>8.3f} adf.r={row['adf.r']:>5.3f}\n"

            # Check minimum distance between two points. Minimum distance is 0.0165 angstroms (in ADF).
            print(f"  Checking if there are atoms too close...")
            pqres_data = pqres_data[['x','y','z','q_fitt']].copy()
            distances = pdist(pqres_data[['x','y','z']].values)
            dist_matrix = squareform(distances)

            pq_data = pqres_data.copy()

            pqres_data.reset_index(drop=True, inplace=True)
            pq_data.reset_index(drop=True, inplace=True)

            # Find points close to each other
            droppoints = set()

            for i in range(dist_matrix.shape[0]):
                if i not in droppoints:  # Check if the point hasn't already been flagged for deletion
                    for j in range(i+1, dist_matrix.shape[1]):
                        if dist_matrix[i, j] <= 0.0165:
                            # Add the charge of j to i
                            pq_data.at[i, 'q_fitt'] += pqres_data.at[j, 'q_fitt']
                            # Flag j for deletion
                            droppoints.add(j)

            # Drop rows that have been flagged
            pq_data = pq_data.drop(index=droppoints)
            pq_data.reset_index(drop=True, inplace=True)

            numdrop = len(pqres_data) - len(pq_data)

            if numdrop != 0:
                print(f"  {numdrop} point(s) were substracted from the point charges for being too close to the others.")
            else:
                print("  No points are too close to the others.")

            pointchargesdf = pq_data.copy()
            pq_data = pq_data.drop(columns=['q_fitt'])
            pq_data.insert(0, 'atom', 'Ar')

            with open('pointcharges.xyz', 'w') as f:
                f.write(f"  {len(pq_data)}\n")
                f.write(f"   Point charges XYZ. Argon atoms were placed in order to visualize them, and q_fitt was drop it. q_fitt = {pqdf_sel['q_fitt'].iloc[0]:>7.6f}.\n")
                for _, row in pq_data.iterrows():
                    f.write(f"{row['atom']:<5} {row['x']:>10.5f} {row['y']:>10.5f} {row['z']:>10.5f}\n")

            adf_pointcharges = ""
            for _, row in pointchargesdf.iterrows():
                adf_pointcharges += f"{row['x']:>8.3f} {row['y']:>8.3f} {row['z']:>8.3f} {row['q_fitt']:>8.6f}\n"
            
            ADF_input(ref, adf_atomlist, solv_name, adf_ghost, adf_pointcharges)
            os.rename(f"./{ref}_DESC.in", f"../{ref}_DESC.in")

    # Create a logfile
    filename = f"logfile_{ATNAME}_{RESNAME}"

    content = (
        "Trajectory analysis done!\n"
        f"Atom Name: {ATNAME}\n"
        f"Residue: {RESNAME}\n"
        f"Average number of atoms per snapshot: {aps:.3f}\n"
        f"Integration distance: {DIST:.3f}\n"
        f"Highest peak distance: {HPD:.3f}\n"
        f"{ref} radius: {r_ref:.3f}\n"
        f"v-Rg of {RESNAME}: {mean_vRg:.3f} ± {std_vRg:.3f}"
    )
    with open(filename, 'w') as f:
        f.write(content)

    os.chdir("..") 
