import pandas as pd

def sequence_with_positivity(protein_data):
    '''
    protein_data: dataframe with single row
    
    '''
    df = pd.DataFrame([x for x in protein_data['sequence'].values[0]], columns=['residue']) 
    
    df['positivity'] = 0
    positive_sites = eval(protein_data['oglcnac sites'].values[0])
    for site in positive_sites:
        df.loc[site - 1, 'positivity'] = 1    
        
    return df

def make_window(sequence:str, idx:int, window_range = (-10,10)):
    assert window_range[0] < 0, 'window_range must start with negative value'
    assert window_range[1] > 0, 'window_range must end with positive value'
    assert idx >= 0, 'index must be over 0'
    assert idx <= len(sequence) - 1, f'index must be equal to or less than {len(sequence) - 1}'
    
    if idx < - window_range[0]:
        window_str = 'O'*(- window_range[0] - idx)
        window_str += sequence[:idx]
        window_str += f'"{sequence[idx]}"'
        window_str += sequence[idx + 1 : idx + window_range[1] + 1]
        
    elif idx > len(sequence) - (window_range[1] + 1):
        window_str = sequence[idx + window_range[0] : idx]
        window_str += f'"{sequence[idx]}"'
        window_str += sequence[idx + 1 :]
        window_str += 'O'*(window_range[1] + 1 - len(sequence) + idx)
        
    else:
        window_str = sequence[idx + window_range[0] : idx]
        window_str += f'"{sequence[idx]}"'
        window_str += sequence[idx + 1 : idx + window_range[1] + 1]
        
    return window_str

mauri_properties = {
          # size         polarity
    'A' : ['very_small', 'aliphatic', 'uncharged', 'nonpolar'],
    'R' : ['long'      , 'basic'    , 'positive' , 'polar'],
    'N' : ['normal'    , 'amide'    , 'uncharged', 'polar'],
    'D' : ['normal'    , 'acidic'   , 'negative' , 'polar'],
    'C' : ['small'     , 'sulfur'   , 'uncharged', 'nonpolar'],
    'Q' : ['normal'    , 'amide'    , 'uncharged', 'polar'],
    'E' : ['normal'    , 'acidic'   , 'negative' , 'polar'],
    'G' : ['glycine'   , 'glycine'  , 'uncharged', 'nonpolar'],
    'H' : ['aromatic'  , 'basic'    , 'positive' , 'polar'],
    'I' : ['small'     , 'aliphatic', 'uncharged', 'nonpolar'],
    'L' : ['small'     , 'aliphatic', 'uncharged', 'nonpolar'],
    'K' : ['long'      , 'basic'    , 'positive' , 'polar'],
    'M' : ['normal'    , 'sulfur'   , 'uncharged', 'nonpolar'],
    'F' : ['aromatic'  , 'aromatic' , 'uncharged', 'nonpolar'],
    'P' : ['proline'   , 'aliphatic', 'uncharged', 'nonpolar'],
    'S' : ['small'     , 'hydroxyl' , 'uncharged', 'polar'],
    'T' : ['small'     , 'hydroxyl' , 'uncharged', 'polar'],
    'W' : ['aromatic'  , 'aromatic' , 'uncharged', 'nonpolar'],
    'Y' : ['aromatic'  , 'aromatic' , 'uncharged', 'nonpolar'],
    'V' : ['very_small', 'aliphatic', 'uncharged', 'nonpolar'],    
}

def mauri_side(window, side_num, col=0): 
    '''
    col = 0: size, 1: chemical, 2: charge, 3: polarity
    '''
    left, center, right = window.split('"')[0], window.split('"')[1], window.split('"')[2]
    if side_num < 0:
        return mauri_properties.get(left[side_num], 'O'*4)[col]
    elif side_num > 0:
        return mauri_properties.get(right[side_num - 1], 'O'*4)[col]
    else:
        return mauri_properties.get(center, 'O'*4)[col]
    
def mauri_npa(window): # count non-polar aliphatic subsite -3 to -1
    count = 0
    left = window.split('"')[0]
    for aa in left[-3:]:
        aa_prop = mauri_properties.get(aa, 'OOOO')
        if aa_prop[1] == 'aliphatic':
            count += 1
    return count

def mauri_ppo(window): # count polar positive at subsite -7 to -5
    count = 0
    left = window.split('"')[0]
    for aa in left[-7:-4]:
        aa_prop = mauri_properties.get(aa, 'OOOO')
        if aa_prop[2] == 'positive':
            count += 1
    return count

def mauri_st(window): # count serine and threonine within the window
    count = 0
    for aa in window:
        if aa == 'S' or aa == 'T':
            count += 1
    return count

def is_proline(window): # check if proline exists at subsite +1
    right = window.split('"')[2]
    return 1 if right[0] == 'P' else 0

def ss_angle(phi, psi):
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
    