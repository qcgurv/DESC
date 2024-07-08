def getelement(atom_str):
    if len(atom_str) > 1 and atom_str[1].isalpha():
        return atom_str[0] + atom_str[1].lower()
    else:
        return atom_str[0]


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