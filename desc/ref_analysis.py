import numpy as np
import pandas as pd
from .geometry import com_calc, ref_radius
from .utils import getelement

def analyze_reference(maindf, ref, atomic_weights):
    print(f"  Looking for {ref}")
    ref_data = maindf[maindf['residue'] == ref]
    natoms_ref = ref_data['index'].nunique()

    # COM of REF
    ref_group = ref_data.groupby(['frame'])
    ref_coms = []
    ref_radii = []

    for frame, group in ref_group:
        ref_com_frame = com_calc(group[['atom', 'x', 'y', 'z']].to_numpy(), atomic_weights)
        ref_coms.append(ref_com_frame)

        ref_coord = group[['x', 'y', 'z']].to_numpy()
        r_ref_frame = ref_radius(ref_coord)
        ref_radii.append(r_ref_frame)

    ref_com = np.mean(ref_coms, axis=0)  # Average Center of Mass of REF
    r_ref = np.mean(ref_radii)  # Average Radius of REF

    # Average structure of REF
    atomdata = ref_data[['index', 'atom']]
    xyzdata = ref_data[['index', 'x', 'y', 'z']]
    avgxyz = xyzdata.groupby('index').mean().reset_index()

    avgref = pd.merge(atomdata.drop_duplicates(), avgxyz, on='index')

    xyz_ref = f"{ref}.xyz"
    with open(xyz_ref, "w") as xyzfile:
        xyzfile.write(f"{natoms_ref}\n")
        xyzfile.write(f"Average coordinates of {ref} residue\n")
        
        for _, row in avgref.iterrows():
            elemref = getelement(row['atom'])
            x, y, z = float(row['x']), float(row['y']), float(row['z'])
            xyzfile.write(f"{elemref:<2s} {x:>8.3f} {y:>8.3f} {z:>8.3f}\n")
    print(f"  Average structure written to {xyz_ref}")

    # Store xyz_ref in adf_atomlist for ADF input
    with open(xyz_ref, 'r') as file:
        xyz_lines = file.readlines()[2:]

    adf_atomlist = ''.join(xyz_lines)
    
    return ref_com, r_ref, adf_atomlist
