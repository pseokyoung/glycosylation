import pandas as pd 
import numpy as np

# for_onehot = { # column_name : classes
#     # for input variables
#     'residue' : ['A', 'R', 'N', 'D', 'C',
#                  'E', 'Q', 'G', 'H', 'I',
#                  'L', 'K', 'M', 'F', 'P',
#                  'S', 'T', 'W', 'Y', 'V'],
    
#     # for output variables
#     'positivity' : [0, 1]
# }

def get_onehots(dataframe, columns=[], for_onehot={}):
    df_col = dataframe[columns].copy()
    df_not = dataframe.loc[:,~dataframe.columns.isin(columns)].copy()
    
    def is_equal(x, key):
        return 1 if x == key else 0      

    for column in columns:
        for key in for_onehot[column]:
            df_col[f'{column}_{key}'] = df_col[column].apply(lambda x: is_equal(x, key))

    df_col = df_col.drop(columns=columns)
    
    return pd.concat([df_col, df_not], axis=1)

def get_window(data, idx, window_size):
    start_point = idx - window_size
    end_point   = idx + window_size
    
    # Adjust start and end index to prevent outbound
    start_idx = max(0, start_point)
    end_idx   = min(len(data) - 1, end_point)
    
    # Extract values within the window
    data_window = data.iloc[start_idx : end_idx + 1].copy()
    
    # Create a DataFrame filled with zeros
    zero_frame = pd.DataFrame(np.zeros((end_point - start_point + 1, data.shape[ 1 ])), columns = data.columns)
    
    # Copy the data into the appropriate location in the zero frame
    if start_point < 0:
        zero_frame.iloc[ - start_point : ] = data_window
        
    elif end_point > len(data) - 1:
        zero_frame.iloc[ : - (end_point-len(data) + 1)] = data_window
    
    else:
        zero_frame = data_window
        
    return zero_frame

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

def data_scaling(train_data, test_data):
    if len(train_data.shape) > 2:
        x_min, x_max = train_data.min(0).min(0), train_data.max(0).max(0)
    else:
        x_min, x_max = train_data.min(0), train_data.max(0)
        
    train_data_sc = (train_data - x_min) / (x_max - x_min)
    test_data_sc  = (test_data - x_min)  / (x_max - x_min)
        
    return train_data_sc, test_data_sc

def upsample_data(train_x, train_y, seed=42):
    np.random.seed(seed)
    pos_indices = np.where(train_y[:,1] == 1)[0]
    neg_indices = np.where(train_y[:,1] == 0)[0]
    pos_upsampled = np.random.choice(pos_indices, size=len(neg_indices), replace=True)
    train_x = np.concatenate([train_x[pos_upsampled], train_x[neg_indices]], axis=0)
    train_y = np.concatenate([train_y[pos_upsampled], train_y[neg_indices]], axis=0)
    shuffle_indices = np.arange(len(train_x))
    np.random.shuffle(shuffle_indices)
    return train_x[shuffle_indices], train_y[shuffle_indices]

