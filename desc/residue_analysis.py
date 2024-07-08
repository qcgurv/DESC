import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.spatial.distance import pdist, squareform
from sklearn_extra.cluster import KMedoids
from .geometry import compute_vRg_frame, avg_vRg, calculate_rdf
from .utils import isbtw
from .file_generation import ADF_input

def create_directory(ATNAME, RESNAME):
    directory = f"{ATNAME}_{RESNAME}"
    os.mkdir(directory)
    os.chdir(directory)
    return directory

def compute_vRg(resid_data, vol_dict):
    vRg_df = compute_vRg_frame(resid_data, vol_dict)
    avg_vRg_df = avg_vRg(vRg_df)
    mean_vRg = avg_vRg_df['v-Rg'].mean()
    std_vRg = avg_vRg_df['v-Rg'].std()
    return mean_vRg, std_vRg

def compute_rdf(res_data, ref_com, box_length):
    bin_r, g_r, bin_width, distances, number_density, bin_edges = calculate_rdf(res_data, ref_com, box_length, bin_width=0.1)
    return bin_r, g_r, bin_width, distances, number_density, bin_edges

def smooth_rdf(g_r):
    g_r_smoothed = savgol_filter(g_r, window_length=11, polyorder=2)
    g_r_fd = savgol_filter(g_r, window_length=11, polyorder=2, deriv=1)
    return g_r_smoothed, g_r_fd

def find_rdf_peaks(g_r_smoothed, bin_width):
    h = 0.5 * g_r_smoothed.max()
    dist = 10 * bin_width
    prom = np.std(g_r_smoothed)
    peaks, _ = find_peaks(g_r_smoothed, height=h, distance=dist, prominence=prom)
    return peaks

def integrate_rdf(bin_r, g_r_fd, peaks):
    res_peak_max_r = []
    res_integration_distances = []
    for peak_index in peaks:
        res_peak_max_r.append(bin_r[peak_index])
        for i in range(peak_index + 1, len(g_r_fd) - 1):
            if g_r_fd[i - 1] < 0 and g_r_fd[i] >= 0:
                res_integration_distances.append(bin_r[i])
                break
        else:
            res_integration_distances.append(None)
    return res_peak_max_r, res_integration_distances

def filter_residue_data(res_data, distances, DIST):
    res_data = res_data.copy()
    res_data['dist'] = distances
    intres_data = res_data[res_data['dist'] <= DIST]
    return intres_data

def clustering(intres_data, CHARGE, eps, val, mean_vRg):
    near = intres_data[['x','y','z']].to_numpy()
    kmed = KMedoids(n_clusters=val, method='alternate', init='heuristic', random_state=42, max_iter=10000).fit(near)
    klabels = kmed.labels_
    intres_data = intres_data.copy()
    intres_data['Cluster'] = klabels
    cluster = kmed.cluster_centers_

    unique_frames = intres_data['frame'].unique()
    pqframes = 100
    if len(unique_frames) >= pqframes:
        sel_frames = pd.Series(unique_frames).sample(n=pqframes)
    else:
        raise ValueError("Not enough unique frames to sample from.")
    
    pqdf_sel = intres_data[intres_data['frame'].isin(sel_frames)].copy()
    N_tot = len(pqdf_sel)
    pqdf_sel['q_fitt'] = (CHARGE * val / N_tot) * (0.025 + 1 / eps)
    pqres_data = pqdf_sel[['x','y','z','q_fitt']]

    clusterdf = pd.DataFrame(cluster, columns=['x', 'y', 'z'])
    for _, row in clusterdf.iterrows():
        pqres_data = pqres_data[~((pqres_data['x'] == row['x']) &
                                (pqres_data['y'] == row['y']) &
                                (pqres_data['z'] == row['z']))] 
    ghostdf = clusterdf.copy()
    ghostdf.insert(0, 'atom', 'Gh.H')
    ghostdf['adf.r'] = mean_vRg
    clusterdf.insert(0, 'atom', 'He')

    adf_ghost = ""
    for _, row in ghostdf.iterrows():
        adf_ghost += f"{row['atom']:<2s} {row['x']:>8.3f} {row['y']:>8.3f} {row['z']:>8.3f} adf.r={row['adf.r']:>5.3f}\n"

    return intres_data, pqdf_sel, pqres_data, clusterdf, ghostdf, adf_ghost

def save_pointcharges(pqres_data, pqdf_sel, val):
    pqres_data.reset_index(drop=True, inplace=True)
    pq_data = pqres_data.copy()
    pq_data.reset_index(drop=True, inplace=True)

    droppoints = set()
    dist_matrix = squareform(pdist(pq_data[['x', 'y', 'z']].values))
    for i in range(dist_matrix.shape[0]):
        if i not in droppoints:
            for j in range(i+1, dist_matrix.shape[1]):
                if dist_matrix[i, j] <= 0.0165:
                    pq_data.at[i, 'q_fitt'] += pqres_data.at[j, 'q_fitt']
                    droppoints.add(j)

    pq_data = pq_data.drop(index=droppoints).reset_index(drop=True)
    pq_data.insert(0, 'atom', 'Ar')

    pqname = f"pointcharges_{val}.xyz"
    with open(pqname, 'w') as f:
        f.write(f"{len(pq_data)}\n")
        f.write(f"Point charges XYZ. Argon atoms were placed in order to visualize them, and q_fitt was dropped. q_fitt = {pqdf_sel['q_fitt'].iloc[0]:>7.6f}.\n")
        for _, row in pq_data.iterrows():
            f.write(f"{row['atom']:<5} {row['x']:>10.5f} {row['y']:>10.5f} {row['z']:>10.5f}\n")

    adf_pointcharges = ""
    for _, row in pq_data.iterrows():
        adf_pointcharges += f"{row['x']:>8.3f} {row['y']:>8.3f} {row['z']:>8.3f} {row['q_fitt']:>8.6f}\n"

    return adf_pointcharges

def weight(a, b, c):
    x = ([[1, 1], [a, b]])
    y = ([1, c])
    w = np.linalg.solve(x, y)
    return w

def write_logfile(ATNAME, RESNAME, aps, DIST, HPD, mean_vRg, std_vRg, solv_name, near_aps=None, far_aps=None, weights=None):
    filename = f"logfile_{ATNAME}_{RESNAME}"
    content = (
        "Trajectory analysis done!\n"
        f"Atom Name: {ATNAME}\n"
        f"Residue: {RESNAME}\n"
        f"Solvent: {solv_name}\n"
        f"Integration distance: I = {DIST:.3f}\n"
        f"Highest peak distance : 1st = {HPD:.3f}\n"
        f"v-Rg of {RESNAME}: {mean_vRg:.3f} ± {std_vRg:.3f}\n"
        f"Average number of counterions per snapshot: {aps:.3f}\n"
    )
    if weights is not None:
        content += (
        f"N={near_aps} weight: {weights[0]:.3f}, N={far_aps} weight: {weights[1]:.3f}\n"
        )

    with open(filename, 'w') as f:
        f.write(content)
    os.chdir("..")


def process_residues(maindf, ref_com, box_length, vol_dict, atomic_weights, eps, ref, r_ref, adf_atomlist, solv_name, atname, resname, qform):
    if isinstance(atname, str):
        atnames = atname.split()
    else:
        atnames = atname

    if isinstance(resname, str):
        resnames = resname.split()
    else:
        resnames = resname

    if isinstance(qform, str):
        qforms = [int(q) for q in qform.split()]
    else:
        qforms = [int(q) for q in qform]

    for ATNAME, RESNAME, CHARGE in zip(atnames, resnames, qforms):
        print(f"  Looking for {ATNAME}-{RESNAME}")
        
        # Create and move into directory
        create_directory(ATNAME, RESNAME)

        # Residue data
        resid_data = maindf[maindf['residue'] == RESNAME]
        res_data = maindf[(maindf['atom'] == ATNAME) & (maindf['residue'] == RESNAME)]
        natoms_res = res_data['residue_number'].nunique()
        
        # Volume-weighted Radius of Gyration
        mean_vRg, std_vRg = compute_vRg(resid_data, vol_dict)
        print(f"  v-Rg of {RESNAME}: {mean_vRg:.3f} ± {std_vRg:.3f}")

        # Radial Distribution Function between COM(REF) and ATNAME
        bin_r, g_r, bin_width, distances, number_density, bin_edges = compute_rdf(res_data, ref_com, box_length)
        rdfdf = pd.DataFrame({"Distance": bin_r, "g(r)": g_r})
        rdf_name = f'rdf_{ref}_{ATNAME}_{RESNAME}.dat'
        os.rename('rdf.dat', rdf_name)

        # Smooth RDF and compute first derivative
        g_r_smoothed, g_r_fd = smooth_rdf(g_r)

        # Find peaks
        peaks = find_rdf_peaks(g_r_smoothed, bin_width)
        res_peak_max_r, res_integration_distances = integrate_rdf(bin_r, g_r_fd, peaks)

        DIST = np.max(res_integration_distances)
        HPD = np.max(res_peak_max_r)
        print(f"  Highest Peak Distance: {HPD:.2f} angstrom")
        print(f"  Total Integration Distance: {DIST:.2f} angstrom")

        intres_data = filter_residue_data(res_data, distances, DIST)
        filt_at = len(intres_data)
        int_ss = int(len(res_data) / natoms_res)
        aps = filt_at / int_ss
        print(f"  Clustering...")

        print_once = False # Print average coordination only once
        (near_aps, far_aps), chkaps = isbtw(aps)
        if chkaps:
            for val in [near_aps, far_aps]:
                if not print_once:
                    if val == 0:
                        continue

                    intres_data, pqdf_sel, pqres_data, clusterdf, ghostdf, adf_ghost = clustering(intres_data, CHARGE, eps, val, mean_vRg)
                    adf_pointcharges = save_pointcharges(pqres_data, pqdf_sel, val)
                    ADF_input(ref, adf_atomlist, solv_name, ghostdf, adf_pointcharges, val)
                    os.rename(f"./{ref}_DESC_{val}.in", f"../{ref}_DESC_{val}.in")

                    clustname = f"ghostatoms_{val}.xyz"
                    with open(clustname, 'w') as f:
                        f.write(f"{len(clusterdf)}\n")
                        f.write("Ghost atoms XYZ. Helium atoms were placed instead of Gh.H to be able to visualize them.\n")
                        for _, row in clusterdf.iterrows():
                            f.write(f"{row['atom']:<5} {row['x']:>10.5f} {row['y']:>10.5f} {row['z']:>10.5f}\n")

                    print(f"  There are {aps:.2f} {ATNAME}-{RESNAME} atoms in average coordinating the solute.")
                    print_once = True

            # Calculate weight of each configuration
            weights = weight(a=near_aps, b=far_aps, c=aps)
            #Create a logfile
            write_logfile(ATNAME, RESNAME, aps, DIST, HPD, mean_vRg, std_vRg, solv_name, near_aps, far_aps, weights)
            
        else:
            if round(aps) == 0:
                print(" -- DONE!")
                print(f"  NOTE: THERE ARE NO {RESNAME} NEAR {ref}. NO CLUSTER ANALYSIS HAS BEEN PERFORMED.")
                break       
            else:     
                val = round(aps)
                intres_data, pqdf_sel, pqres_data, clusterdf, ghostdf, adf_ghost = clustering(intres_data, CHARGE, eps, val, mean_vRg)
                adf_pointcharges = save_pointcharges(pqres_data, pqdf_sel, val)
                ADF_input(ref, adf_atomlist, solv_name, adf_ghost, adf_pointcharges, val)
                os.rename(f"./{ref}_DESC_{val}.in", f"../{ref}_DESC_{val}.in")

                with open('ghostatoms.xyz', 'w') as f:
                    f.write(f"{len(clusterdf)}\n")
                    f.write("Ghost atoms XYZ. Helium atoms were placed instead of Gh.H to be able to visualize them.\n")
                    for _, row in clusterdf.iterrows():
                        f.write(f"{row['atom']:<5} {row['x']:>10.5f} {row['y']:>10.5f} {row['z']:>10.5f}\n")

                print(f"  There are {aps:.2f} {ATNAME}-{RESNAME} atoms in average coordinating the solute.")
        
            #Create a logfile
            write_logfile(ATNAME, RESNAME, aps, DIST, HPD, mean_vRg, std_vRg, solv_name)