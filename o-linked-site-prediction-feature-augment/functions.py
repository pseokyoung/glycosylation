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

