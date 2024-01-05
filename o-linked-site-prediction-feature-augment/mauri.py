import numpy as np

def mauli_features(properties:object):
    dataset = properties.all_properties()
    
    if len(dataset):
        sequence = dataset['SEQ'].sum()
        dataset['window'] = dataset['#'].apply(lambda x: make_window(sequence=sequence, index=x, window=SEQUENCE_WINDOW))
        dataset['marker'] = dataset['#'].apply(lambda x: make_window(sequence=sequence, index=x, window=SEQUENCE_WINDOW, marking=True))
    
        for side in range(SIDE_WINDOW[0], SIDE_WINDOW[1] + 1):
            if side != 0:
                dataset[f'side_{side}'] = dataset.marker.apply(lambda x: side_chain(x, side=side))

        dataset[f'nAli{ALIPHATIC_WINDOW}'] = dataset.apply(lambda x: count_nonpolar_aliphatic(make_window(sequence, x['#'], ALIPHATIC_WINDOW)), axis=1)
        dataset[f'nPos{POSITIVE_CHARGE_WINDOW}'] = dataset.apply(lambda x: count_positively_charged(make_window(sequence, x['#'], POSITIVE_CHARGE_WINDOW)), axis=1)
        dataset['nS/nT'] = dataset.window.apply(count_SorT)
        dataset[f'is_proline({PROLINE_AFTER})'] = dataset.marker.apply(lambda x: is_proline_after(x, after=PROLINE_AFTER))
        dataset['phi_psi'] = dataset.apply(lambda x: angle_type(x['Phi'], x['Psi']), axis=1)
        
        return dataset
    
    else:
        return []

DTYPES = {'positivity': 'int64', 
          f'nAli{ALIPHATIC_WINDOW}': 'int64',
          f'nPos{POSITIVE_CHARGE_WINDOW}': 'int64',
          'nS/nT': 'int64',
          f'is_proline({PROLINE_AFTER})': 'int64'}

amino_acid = {"A":1, "R":2, "N":3, "D":4, "C":5, 
              "E":6, "Q":7, "G":8, "H":9, "I":10, 
              "L":11, "K":12, "M":13, "F":14, "P":15, 
              "S":16, "T":17, "W":18, "Y":19, "V":20}

'''
1)  Alanine (Ala, A)
2)  Arginine (Arg, R)
3)  Asparagine (Asn, N)
4)  Aspartic acid (Asp, D)
5)  Cysteine (Cys, C)
6)  Glutamic acid (Glu, E)
7)  Glutamine (Gln, Q)
8)  Glycine (Gly, G)
9)  Histidine (His, H)
10) Isoleucine (Ile, I)
11) Leucine (Leu, L)
12) Lysine (Lys, K)
13) Methionine (Met, M)
14) Phenylalanine (Phe, F)
15) Proline (Pro, P)
16) Serine (Ser, S)
17) Threonine (Thr, T)
18) Tryptophan (Trp, W)
19) Tyrosine (Tyr, Y)
20) Valine (Val, V)
'''

def make_window(sequence:str, index:int, window:tuple=(-10,10), marking=False):
    assert index > 0, 'index must be over 0'
    assert index <= len(sequence), f'index must be equal to or less than {len(sequence)}'
    
    index -= 1
    start = index+window[0]
    end   = index+window[1]
    len_seq = len(sequence)
    
    if start >= len_seq or end < 0:
        return ''
    
    elif start < 0:
        window_str = 'O'*(-start)
        window_str += sequence[0:end+1]
        
    elif end >= len_seq:
        window_str = sequence[start:]
        window_str += 'O'*(end-len_seq+1)
        
    else:
        window_str = sequence[start:end+1]
        
    if marking:
        temp = window_str
        window_str = temp[:-window[0]]
        window_str += f"'{temp[-window[0]]}'"
        window_str += temp[-window[1]:]

    return window_str


def chain_type(letter):
    if (letter == "G"): #1
        return 'gly' 
    elif (letter == "V" or letter == "A"): #2 Val, Ala
        return 'very_small'
    elif (letter == "S" or letter == "I" or letter == "L" or letter == "T" or letter == "C"): #3 Ser, Thr, Ile, Leu, Cys
        return 'small'
    elif (letter == "D" or letter == "E" or letter == "N" or letter == "Q" or letter == "M"): #4 Asp, Asn, Glu, Gln, Met
        return 'normal'
    elif (letter == "R" or letter == "K"): #5 Arg, Lys
        return 'long'
    elif (letter == "F" or letter == "W" or letter == "Y" or letter == "H"): #6 Phe, Trp, Tyr, His
        return 'cycle'
    elif (letter == "P"): #7
        return 'pro'
    else:
        return 'None' #0
    
    
def side_chain(window_str_with_mark, side):
    center_idx = window_str_with_mark.index("'") + 1
    target_idx = center_idx + side + 1*np.sign(side)
    
    return chain_type(window_str_with_mark[target_idx])
    
    
def count_nonpolar_aliphatic(window_str):
    nA = window_str.count("A")
    nV = window_str.count("V")
    nL = window_str.count("L")
    nI = window_str.count("I")
    nP = window_str.count("P")
    
    return nA + nV + nL + nI + nP


def count_positively_charged(window_str):
    nR = window_str.count("R")
    nK = window_str.count("K")
    nH = window_str.count("H")
    
    return nR + nK + nH


def count_SorT(window_str):
    nS = window_str.count("S")
    nT = window_str.count("T")
    
    return nS + nT


def is_proline_after(window_str_with_mark, after=1): 
    """
    Check if there is a proline residue after the marked position in a given amino acid sequence.

    Args:
        window_str_with_mark (str): Amino acid sequence with a marked position using a single quote (').
        after (int, optional): Number of positions after the marked position to check for proline. Defaults to 1.

    Returns:
        int: 1 if there is a proline residue after the marked position, 0 otherwise.
    """
    
    center_idx = window_str_with_mark.index("'") + 1
    target_idx = center_idx + after + 1*np.sign(after)
    
    if 0 <= target_idx < len(window_str_with_mark):
        return int(window_str_with_mark[target_idx] == 'P')

    else:
        return 0
    
    
def angle_type(phi, psi):
    """
    Determine the type of dihedral angle based on phi and psi values.

    Args:
        phi (float): Phi angle in degrees.
        psi (float): Psi angle in degrees.

    Returns:
        str: The type of the dihedral angle ("alpha", "beta", or "other").
    """
    both_phi_range  = (-160, -50)
    alpha_psi_range = (100, 180)
    beta_psi_range  = (-60, 20)
    
    # Check if the given angles are within the alpha region
    if both_phi_range[0] < phi < both_phi_range[1] and alpha_psi_range[0] < psi < alpha_psi_range[1]:
        return "alpha"

    # Check if the given angles are within the beta region
    elif both_phi_range[0] < phi < both_phi_range[1] and beta_psi_range[0] < psi < beta_psi_range[1]:
        return "beta"
    
    else:
        return "other"
    