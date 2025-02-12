import numpy as np
import pandas as pd
from .utils import getelement
from .constants import atomic_weights, vol_dict

def ref_radius(coordinates):
    r_ref = 0
    max_dist = 0
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            dist = np.linalg.norm(coordinates[i] - coordinates[j])
            max_dist = max(max_dist, dist)
            r_ref = max_dist / 2
    return r_ref

def com_calc(data, atomic_weights):
    elements = np.array([getelement(atom) for atom in data[:, 0]])
    coordinates = data[:, 1:4].astype(np.float32)
    atomic_weights_vector = np.array([atomic_weights.get(elem, 1.0) for elem in elements], dtype=np.float32)
    weighted_coordinates = coordinates * atomic_weights_vector[:, np.newaxis]
    total_mass = atomic_weights_vector.sum()
    com = weighted_coordinates.sum(axis=0) / total_mass
    return com

def rog_calc(atoms, vol_dict):
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
        rg = rog_calc(atoms, vol_dict)
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
