from .utils import getelement
from openbabel import openbabel as ob
from rdkit import Chem

# Generate the content of the xyz file
def generate_xyz(df):
    xyz_content = []
    num_atoms = len(df)
    xyz_content.append(f"{num_atoms}")
    xyz_content.append("Generated XYZ file")

    for _, row in df.iterrows():
        xyz_content.append(f"{row['atom']} {row['x']} {row['y']} {row['z']}")

    return "\n".join(xyz_content)

def convert_xyz_to_mol(xyz_content):
    # Initialize Open Babel conversion
    conv = ob.OBConversion()
    conv.SetInAndOutFormats("xyz", "mol")
    
    # Read XYZ string
    mol = ob.OBMol()
    conv.ReadString(mol, xyz_content)
    
    return mol

def obmol_to_rdkit_mol(obmol):
    # Convert OBMol to a MOL string
    conv = ob.OBConversion()
    conv.SetOutFormat("mol")
    mol_str = conv.WriteString(obmol)
    
    # Convert MOL string to RDKit Mol
    rdkit_mol = Chem.MolFromMolBlock(mol_str, sanitize=True, removeHs=True)
    
    if rdkit_mol is None:
        print("Error: Could not convert OBMol to RDKit Mol")
        return None
    
    return rdkit_mol

def generate_smiles_from_xyz(xyz_content):
    # Convert XYZ to OBMol
    obmol = convert_xyz_to_mol(xyz_content)
    
    # Convert OBMol to RDKit Mol
    rdkit_mol = obmol_to_rdkit_mol(obmol)
    
    if rdkit_mol is None:
        return None
    
    # Generate SMILES
    smiles = Chem.MolToSmiles(rdkit_mol, canonical=True)
    
    return smiles

def analyze_solvent(maindf, adf_solvent):
    # First Frame, only to identify the solvent resname
    ff = maindf[maindf['frame'] == maindf['frame'].min()]
    solv_resname = ff['residue'].value_counts().idxmax()
    solvmatch = ff[ff['residue'] == solv_resname].iloc[0]
    residue_number = solvmatch['residue_number']

    filt_ff = ff[(ff['residue_number'] == residue_number) & (ff['residue'] == solv_resname)]
    filt_ff = filt_ff.copy()
    filt_ff['atom'] = filt_ff['atom'].apply(getelement)
    solv_xyz = generate_xyz(filt_ff)
    solv_smiles = generate_smiles_from_xyz(solv_xyz)
 
    if solv_smiles is not None:
        pass
    else:
        print("Could not generate SMILES string for the solvent molecule.")
    
    # Compare the standardized formula with the dictionary
    for name, properties in adf_solvent.items():
        if properties['formula'] == solv_smiles:
            solv_name = name
            eps = properties['eps']
            break
    else:
        print("Solvent formula not found in the ADF solvent database.")
        print("Solvents:")

        solvent_names = list(adf_solvent.keys())
        for i in range(0, len(solvent_names), 4):
            print('{:<25} {:<25} {:<25} {:<25}'.format(
                solvent_names[i],
                solvent_names[i+1] if i+1 < len(solvent_names) else '',
                solvent_names[i+2] if i+2 < len(solvent_names) else '',
                solvent_names[i+3] if i+3 < len(solvent_names) else '',
            ))

        while True:
            solv_name_input = input("Please specify the solvent name (must be in the ADF solvent database): ")
            solv_name = next((name for name in adf_solvent if name.lower() == solv_name_input.lower()), None)
            if solv_name:
                eps = adf_solvent[solv_name]['eps']
                break
            else:
                print(f"WARNING: '{solv_name}' is not in the ADF solvent database. Please enter a valid solvent name.")
                while True:
                    try:
                        eps = float(input(f"Please specify the dielectric constant (eps) for the solvent '{solv_name}': "))
                        solv_name = solv_name_input
                        break
                    except ValueError:
                        print("Invalid input. Please enter a numerical value for the dielectric constant.")
                print(f"Continuing with specified values for solvent '{solv_name}'.")
                break             
                
    return solv_name, eps
