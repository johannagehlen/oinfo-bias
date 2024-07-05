import seaborn as sns 
sns.set_theme()
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

def plot_heatmaps(measure, measure_savename, colorbar_label):
    fig, ax = plt.subplots(1,3, figsize = (20, 7))
    sns.set(font_scale = 2)

    df_redundant = pd.read_csv(f"./simulation_results/dataframes/redundant/{measure}.csv")
    df_independent = pd.read_csv(f"./simulation_results/dataframes/independent/{measure}.csv")
    df_synergistic = pd.read_csv(f"./simulation_results/dataframes/synergistic/{measure}.csv")
    
    df_redundant["N"] = list(range(5, 51, 2))
    df_independent["N"] = list(range(500, 20000, 500))
    df_synergistic["N"] = list(range(500, 20000, 500))

    df_redundant = df_redundant.set_index("N")
    df_independent = df_independent.set_index("N")
    df_synergistic = df_synergistic.set_index("N")

    
    cols_with_k = [i for i in df_redundant.columns if "bins" in i]
    
    df_redundant = df_redundant[cols_with_k]
    df_independent = df_independent[cols_with_k]
    df_synergistic = df_synergistic[cols_with_k]

    vmin = min(df_redundant.min().min(), df_independent.min().min(), df_synergistic.min().min())
    vmax = max(df_redundant.max().max(), df_independent.max().max(), df_synergistic.max().max())

    abs_max = max(abs(vmin), abs(vmax))

    # Create the heatmap 
    if "Standard Deviation" in colorbar_label:   
        vmin = 0
    else:
        vmin = -abs_max
    
    h0 = sns.heatmap(df_redundant, ax=ax[0], cmap="RdBu_r", cbar=False, vmin = vmin, vmax = abs_max)
    h0.invert_yaxis()
    ax[0].set_ylabel("Sample size N", fontsize=30)
    ax[0].set_title("Fully Redundant", fontsize=30)


    h1 = sns.heatmap(df_independent, ax = ax[1], cbar = False, cmap = "RdBu_r", vmin = vmin, vmax = abs_max)
    h1.invert_yaxis()
    ax[1].set_title("Independent", fontsize=30)
    ax[1].set_ylabel("", fontsize=30)

    h2 = sns.heatmap(df_synergistic, ax = ax[2], cmap = "RdBu_r", cbar = False, vmin = vmin, vmax = abs_max)
    h2.invert_yaxis()
    ax[2].set_title("Fully Synergistic", fontsize=30)
    ax[2].set_ylabel("", fontsize=30)

    for i in range(3):
        ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=0)
        ax[i].set_xticklabels([int(item.get_text().split("_")[0]) for item in ax[i].get_xticklabels()])
        ax[i].set_xticks(range(0, 50, 5)) 
        ax[i].set_xticklabels(range(0, 50, 5))

    #fig.subplots_adjust(wspace=0.05)
    fig.text(0.5, -0.02, "Number of bins K", ha='center', fontsize=30)

    #put one color bar for the whole figure 
    cbar_ax = fig.add_axes([0.99, 0.14, 0.03, 0.7])  
    cbar = fig.colorbar(h2.get_children()[0], cax=cbar_ax)
    cbar.set_label(f"{colorbar_label}", fontsize=30)

    plt.subplots_adjust(left=0.07, right=0.9, wspace=0.1)

    fig.tight_layout()

    plt.savefig(f"./simulation_results/heatmaps/{measure_savename}.pdf", dpi = 300, bbox_inches = "tight")
    plt.show


def plot_heatmaps_with_lines(measure, measure_savename, colorbar_label):
    fig, ax = plt.subplots(1, 3, figsize=(20, 7))
    sns.set(font_scale=2)

    df_redundant = pd.read_csv(f"./simulation_results/dataframes/redundant/{measure}.csv")
    df_independent = pd.read_csv(f"./simulation_results/dataframes/independent/{measure}.csv")
    df_synergistic = pd.read_csv(f"./simulation_results/dataframes/synergistic/{measure}.csv")
    
    df_redundant["N"] = list(range(5, 51, 2))
    df_independent["N"] = list(range(500, 20000, 500))
    df_synergistic["N"] = list(range(500, 20000, 500))

    df_redundant = df_redundant.set_index("N")
    df_independent = df_independent.set_index("N")
    df_synergistic = df_synergistic.set_index("N")


    cols_with_k = [i for i in df_redundant.columns if "bins" in i]

    df_redundant = df_redundant[cols_with_k]
    df_independent = df_independent[cols_with_k]
    df_synergistic = df_synergistic[cols_with_k]

    vmin = min(df_redundant.min().min(), df_independent.min().min(), df_synergistic.min().min())
    vmax = max(df_redundant.max().max(), df_independent.max().max(), df_synergistic.max().max())

    abs_max = max(abs(vmin), abs(vmax))

    for i, df in enumerate([df_redundant, df_independent, df_synergistic]):

        # Create the heatmap
        h = sns.heatmap(df, ax=ax[i], cmap="RdBu_r", cbar=False, vmin=-abs_max, vmax=abs_max)
        h.invert_yaxis()

        ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=0)
        ax[i].set_xticklabels([int(item.get_text().split("_")[0]) for item in ax[i].get_xticklabels()])
        ax[i].set_xticks(range(0, 50, 5))
        ax[i].set_xticklabels(range(0, 50, 5))
        
        x_values = np.arange(0, 50, 1)

        # Add lines
        if i == 0:
            y_values_1 = (x_values) / 2
            y_values_2 = (x_values ** 2) / 2
            y_values_3 = (x_values ** 3) / 2
        else:
            y_values_1 = (x_values) / 500
            y_values_2 = (x_values ** 2) / 500
            y_values_3 = (x_values ** 3) / 500


        ax[i].plot(x_values, y_values_1, color='red', linestyle='--', linewidth=3, label='$N = K$')
        ax[i].plot(x_values, y_values_2, color='green', linestyle='--', linewidth=3, label='$N = K^2$')
        ax[i].plot(x_values, y_values_3, color='blue', linestyle='--', linewidth=3, label='$N = K^3$')

        ax[i].set_ylabel("Sample size N", fontsize=30)
        ax[i].set_title(["Fully Redundant", "Independent", "Fully Synergistic"][i], fontsize=30)

        ax[i].legend()


    fig.text(0.5, -0.02, "Number of bins K", ha='center', fontsize=30)

    # put one color bar for the whole figure 
    cbar_ax = fig.add_axes([0.99, 0.14, 0.03, 0.7])
    cbar = fig.colorbar(h.get_children()[0], cax=cbar_ax)
    cbar.set_label(f"{colorbar_label}", fontsize=30)

    plt.subplots_adjust(left=0.07, right=0.9, wspace=0.1)
    
    fig.tight_layout()

    plt.savefig(f"./simulation_results/heatmaps/{measure_savename}.pdf", dpi=300, bbox_inches="tight")
    plt.show()



def plot_yfs_oinfos(sorted_df, ls_oinfos, ls_oinfos_bc):
    sns.histplot(sorted_df["oinfo_bc"], label = r"Distribution of $\hat{\Omega}_{BC'}$" +  " of all YFS triplets", bins = 50, stat = "probability", alpha = 0.5,)
    sns.histplot(sorted_df["oinfo"], label = r"Distribution of $\hat{\Omega}$" +  " of all YFS triplets", bins = 50, stat = "probability", alpha = 0.5, color = "g")
    
    plt.axvline(x = np.mean(ls_oinfos_bc), color = "red", linestyle = "--", label = "Mean Bias-Corrected O-information \n" +  r"$\overline{\hat{\Omega}_{BC'}} = $" + f"{round(np.mean(ls_oinfos_bc), 3)}" + " of simulated independent triplets")
    plt.axvline(x = np.mean(ls_oinfos), color = "orange", linestyle = "--", label = r"Mean Naive O-information $\overline{\hat{\Omega}} = $" + f"{round(np.mean(ls_oinfos), 3)}" + "\n of simulated independent triplets")

    sns.histplot(ls_oinfos, label = r"Distribution of ${\hat{\Omega}}$" + "\n from simulated independent triplets", color = "darkgreen")
    sns.histplot(ls_oinfos_bc, label = r"Distribution of ${\hat{\Omega}_{BC'}}$" + "\n from simulated independent triplets", color = "darkblue")
    
    plt.xlabel("O-information", fontsize = 15)
    plt.xticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
    plt.ylabel("Probability", fontsize = 15)

    #increase fontsize of ticks and labels
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.tick_params(axis='both', which='minor', labelsize=10)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.yscale("log")
    plt.savefig("./yfs_results/oinfo_distribution.png", dpi = 300, bbox_inches='tight')


def plot_yfs_depression_oinfos(sorted_df_depression, ls_oinfos, ls_oinfos_bc):
    sns.histplot(sorted_df_depression["oinfo_bc"], label = r"Distribution of $\hat{\Omega}_{BC'}$ " + "\n of all YFS depression triplets", bins = 50, stat = "probability", alpha = 0.5,)
    #sns.histplot(sorted_df_depression["oinfo"], label = r"Distribution of $\hat{\Omega}$" +  "\n of all YFS depression triplets", bins = 20, stat = "probability", alpha = 0.5, color = "g")
    
    plt.axvline(x = np.mean(ls_oinfos_bc), color = "darkblue", linestyle = "--", label = "Mean Bias-Corrected \n"  r"O-information $\overline{\hat{\Omega}_{BC'}} = $" + f"{round(np.mean(ls_oinfos_bc), 3)}" + "\n of simulated independent triplets")
    #plt.axvline(x = np.mean(ls_oinfos), color = "darkgreen", linestyle = "--", label = r"Mean Naive O-information $\overline{\hat{\Omega}} = $" + f"{round(np.mean(ls_oinfos), 3)}" + "\n from simulated independent triplets")

    plt.xlabel("O-information", fontsize = 15)
    plt.xticks([-0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2])
    plt.ylabel("Probability", fontsize = 15)

    #increase fontsize of ticks and labels
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.tick_params(axis='both', which='minor', labelsize=10)

    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend()
    #plt.yscale("log")
    plt.savefig("./yfs_results/oinfo_distribution_depression.pdf", dpi = 300, bbox_inches='tight')

def plot_slices(bin_number):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for index, i in enumerate(["redundant", "independent", "synergistic"]):
        oinfo_naive = pd.read_csv(f"./simulation_results/dataframes/{i}/df_mean_oinfo.csv")
        oinfo_bc = pd.read_csv(f"./simulation_results/dataframes/{i}/df_mean_oinfo_bc.csv")
        oinfo_true = pd.read_csv(f"./simulation_results/dataframes/{i}/true_oinfo.csv")

        cols_with_k = [i for i in oinfo_true.columns if "bins" in i]

        oinfo_naive = oinfo_naive[cols_with_k]
        oinfo_bc = oinfo_bc[cols_with_k]
        oinfo_true = oinfo_true[cols_with_k]

        if index == 0:
            x_vals = [a*2 for a in list(oinfo_naive.index)]
        else:
            x_vals = [a*500 for a in list(oinfo_naive.index)]

        sns.lineplot(data=oinfo_naive, y=f"{bin_number}_bins", x=x_vals, ax=axes[index], label=r"$\overline{\hat{\Omega}}$")
        sns.lineplot(data=oinfo_bc, y=f"{bin_number}_bins", x=x_vals, ax=axes[index], label=r"$\overline{\hat{\Omega}_{BC'}}$")
        sns.lineplot(data=oinfo_true, y=f"{bin_number}_bins", x=x_vals, ax=axes[index], label=r"$\Omega$")

    axes[0].set_title("Fully Redundant", fontsize=20)
    axes[1].set_title("Independent", fontsize=20)
    axes[2].set_title("Fully synergistic", fontsize=20)

    for ax in axes:
        ax.set_xlabel("N", fontsize=20)
        if ax == axes[0]:
            ax.set_ylabel("O-information", fontsize=20)
        else:
            ax.set_ylabel("", fontsize=20)

        # Increase font size of x and y ticks
        ax.tick_params(axis='both', labelsize=12)

    # remove individual legends, add one legend for all
    axes[0].legend().remove()
    axes[1].legend().remove()
    axes[2].legend().remove()
    axes[2].legend(loc="upper right", fontsize=12)

    plt.tight_layout()

    # decrease space between plots 
    plt.subplots_adjust(wspace=0.3)

    # add more xticks 
    axes[0].set_xticks(list(range(0, 51, 10)))
    axes[1].set_xticks(list(range(500, 25000, 5000)))
    axes[2].set_xticks(list(range(500, 25000, 5000)))

    # rotate xticks 
    axes[0].tick_params(axis='x', rotation=90)
    axes[1].tick_params(axis='x', rotation=90)
    axes[2].tick_params(axis='x', rotation=90)

    # remove "N" label from xaxis, and do one joint "sample size N" label 
    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    axes[2].set_xlabel("")

    # add one joint "sample size N" label
    fig.text(0.5, -0.05, 'Sample size N', ha='center', va='center', fontsize=20)

    # increase line thickness of all lines 
    for ax in axes:
        for line in ax.lines:
            line.set_linewidth(2.0)

    # tight layout in order to avoid overlap of labels and titles
    plt.savefig(f"./simulation_results/slice_plots/{bin_number}.pdf", dpi=300, bbox_inches="tight")
    plt.show()
