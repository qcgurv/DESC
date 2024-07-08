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