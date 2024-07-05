import numpy as np 
import pandas as pd 
from helper_functions_oinfo import compute_oinfo
import math

def discretise(data, k, method='quantile'):
    """ 
    inputs:
        data = the numpy nd_array of the data to discretise
        k = number of bins to discretise into 
        method = equal_width or quantile 
    returns: 
        the discretised data as a pandas df
    """
    # use np.cut and np.qcut instead, and compare. 

    data_df = pd.DataFrame(data)
    number_cols = data_df.shape[1]
    number_obs = data_df.shape[0]

    discretized_data = np.zeros((number_obs, number_cols), dtype=int)
    
    if method == 'equal_width':
        if number_cols == 1:
            discretized_data = pd.cut(data, k, labels=False)
        else:
            for i in range(number_cols):
                discretized_data[:, i] = pd.cut(data[:, i], k, labels=False)

    elif method == "quantile":
        if number_cols == 1:
            discretized_data = pd.qcut(data.T.flatten(), k, labels=False, duplicates='drop')
            discretized_data = np.array(discretized_data)[np.newaxis].T

        else:
            for i in range(number_cols):
                discretized_data[:, i] = pd.qcut(data[:, i], k, labels=False, duplicates='drop')

    return discretized_data

def create_dataset(distribution, method, n, k):
    """
    Generate continuous data
    """

    if distribution == "independent":
        data = np.random.uniform(-3, 3, (n, 3))
        return discretise(data, k, method)
    
    elif distribution == "redundant":
        data = np.random.uniform(-3, 3, (n, 1))
        data_disc = discretise(data, k, method)
        return np.concatenate((data_disc, data_disc, data_disc), axis=1)
    
    elif distribution == "synergistic":
        data = np.random.uniform(-3, 3, (n, 2))
        data_disc = discretise(data, k, method)
        X_n = np.sum(data_disc, axis=1)
        X_n_mod = np.mod(X_n, k)
        X_n_mod = X_n_mod.reshape(-1, 1)
        return np.concatenate((data_disc, X_n_mod), axis=1)

def vary_n(param_dict, k):

    ls_oinfo = []
    ls_oinfo_bc = []

    distribution = param_dict["distribution"]
    N_range = param_dict["N_range"]
    method = param_dict["method"]

    for n in N_range:
        # create data with n samples and k bins
        data = create_dataset(distribution, method, n, k)
        data = pd.DataFrame(data)
        oinfo, oinfo_bc = compute_oinfo(data)
        # calculate the oinfo and oinfo_bc for the dataset for the given n and k
        ls_oinfo.append(oinfo)
        ls_oinfo_bc.append(oinfo_bc)
    
    return ls_oinfo, ls_oinfo_bc

def vary_k(param_dict):
    N_range = param_dict["N_range"]
    K_range = param_dict["K_range"]

    df_oinfo = pd.DataFrame({"N": N_range})
    df_oinfo_bc = pd.DataFrame({"N": N_range})

    df_oinfo = df_oinfo.set_index("N")
    df_oinfo_bc = df_oinfo_bc.set_index("N")

    for k in K_range:        
        ls_oinfo, ls_oinfo_bc = vary_n(param_dict, k)

        df_oinfo[f"{k}_bins"] = ls_oinfo
        df_oinfo_bc[f"{k}_bins"] = ls_oinfo_bc

    return df_oinfo, df_oinfo_bc

def run_all_trials(param_dict, yfs_data = False):
    distribution = param_dict["distribution"]
    n_range = param_dict["N_range"]
    number_trials = param_dict["trials"]
    
    dict_results = {}
    
    ls_dfs_oinfo = []
    ls_dfs_oinfo_bc = []

    for trial in range(number_trials):
        df_oinfo, df_oinfo_bc = vary_k(param_dict)
        ls_dfs_oinfo.append(df_oinfo)
        ls_dfs_oinfo_bc.append(df_oinfo_bc)

    dict_results["df_mean_oinfo"] = pd.concat(ls_dfs_oinfo).groupby(level=0).mean()
    dict_results["df_std_oinfo"] = pd.concat(ls_dfs_oinfo).groupby(level=0).std()
    dict_results["df_mean_oinfo_bc"] = pd.concat(ls_dfs_oinfo_bc).groupby(level=0).mean()
    dict_results["df_std_oinfo_bc"] = pd.concat(ls_dfs_oinfo_bc).groupby(level=0).std()

    for key, df in dict_results.items():
        df["N"] = n_range
        df = df.set_index("N")
        if yfs_data:
            df.to_csv(f"./simulation_results/dataframes/yfs/{distribution}/{key}.csv")
        else:
            df.to_csv(f"./simulation_results/dataframes/{distribution}/{key}.csv")
    
    return ls_dfs_oinfo, ls_dfs_oinfo_bc

def create_true_oinfo_df(param_dict):
    distribution = param_dict["distribution"]
    n_range = param_dict["N_range"]
    k_range = param_dict["K_range"]
    
    df_true_oinfo = pd.DataFrame()
    df_true_oinfo["N"] = n_range

    if distribution == "independent":
        for k in k_range:
            df_true_oinfo[f"{k}_bins"] = 0    
    
    elif distribution == "redundant":
        for k in k_range:
            df_true_oinfo[f"{k}_bins"] = math.log2(k)
    
    elif distribution == "synergistic":
        for k in k_range:
            df_true_oinfo[f"{k}_bins"] = -math.log2(k)
    
    df_true_oinfo.to_csv(f"./simulation_results/dataframes/{distribution}/true_oinfo.csv")

def create_oinfo_bias_df(distribution):
    df_true_oinfo = pd.read_csv(f"./simulation_results/dataframes/{distribution}/true_oinfo.csv")
    df_naive_oinfo = pd.read_csv(f"./simulation_results/dataframes/{distribution}/df_mean_oinfo.csv")

    cols_with_k = [i for i in df_true_oinfo.columns if "bins" in i]

    df_oinfo_bias = df_true_oinfo[cols_with_k] - df_naive_oinfo[cols_with_k]
    df_oinfo_bias.set_index(df_true_oinfo["N"])

    df_oinfo_bias.to_csv(f"./simulation_results/dataframes/{distribution}/oinfo_bias.csv")


def create_error_in_bc_df(dict_param_settings):
    distribution = dict_param_settings["distribution"]

    oinfo_bias = pd.read_csv(f"./simulation_results/dataframes/{distribution}/oinfo_bias.csv")
    
    true_oinfo = pd.read_csv(f"./simulation_results/dataframes/{distribution}/true_oinfo.csv")

    bc_oinfo = pd.read_csv(f"./simulation_results/dataframes/{distribution}/df_mean_oinfo_bc.csv")
    
    cols_with_k = [i for i in oinfo_bias.columns if "bins" in i]

    bc_oinfo_bias = true_oinfo[cols_with_k] - bc_oinfo[cols_with_k]

    epsilon_dash = abs(bc_oinfo_bias[cols_with_k]) - abs(oinfo_bias[cols_with_k])
    epsilon_dash["N"] = dict_param_settings["N_range"]
    epsilon_dash= epsilon_dash.set_index("N")
    epsilon_dash.to_csv(f"./simulation_results/dataframes/{distribution}/df_epsilon_dash.csv")

    return epsilon_dash