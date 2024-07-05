import math 
import numpy as np 

def entropyd(sx, base=2):
    """Discrete entropy estimator
    sx is a list of samples
    """
    _, count = np.unique(sx, return_counts=True, axis=0)
    # Convert to float as otherwise integer division results in all 0 for proba.
    proba = count.astype(float) / len(sx)
    # Avoid 0 division; remove probabilities == 0.0 (removing them does not change the entropy estimate as 0 * log(1/0) = 0.
    proba = proba[proba > 0.0]
    return np.sum(proba * np.log(1.0 / proba)) / np.log(base)

def approx_oinfo_bias(data_selected):
    """ 
    A function that calculates the Miller-Maddow O-information bias approximation term. 
    triplet: a list of three strings, the names of the columns in the dataframe
    yfs_data: the dataframe with the data

    Returns: the Miller-Maddow O-information bias approximation term
    """
    sample_size = data_selected.shape[0] #number of samples 
    number_variables = data_selected.shape[1] #number of variables

    sum_term = 0
    for i in data_selected.columns:
        K_i = data_selected[i].nunique()
        combination_counts_joint_withouti = data_selected.groupby([a for a in data_selected.columns if a != i]).size()
        K_withouti = len(combination_counts_joint_withouti)
        sum_term += (K_i - K_withouti)

    # joint entropy bias term
    combination_counts_joint = data_selected.groupby(list(data_selected.columns)).size()
    K_n = len(combination_counts_joint)

    # oinfo_bias_term = individual_entropy_bias_term - 2*joint_entropy_bias_term + conditional_entropy_bias_term
    oinfo_bias_term = 1/(2*sample_size*math.log(2)) * (sum_term + (number_variables-2)*K_n - number_variables + 2)
   
    return oinfo_bias_term

def compute_oinfo(data):

    o_info = (len(data.columns) - 2)*entropyd(data)
    
    for j, col in enumerate(data.columns):

        o_info += entropyd(data.loc[:, data.columns == col])
        o_info -= entropyd(data.loc[:, data.columns != col])
    
    bias_term  = approx_oinfo_bias(data)
    o_info_bc = o_info + bias_term

    return o_info, o_info_bc
