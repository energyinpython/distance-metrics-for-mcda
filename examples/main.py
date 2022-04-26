import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from distance_metrics_mcda.mcda_methods import TOPSIS
from distance_metrics_mcda.additions import rank_preferences
from distance_metrics_mcda import correlations as corrs
from distance_metrics_mcda import normalizations as norms
from distance_metrics_mcda import distance_metrics as dists
from distance_metrics_mcda import weighting_methods as mcda_weights


# Functions for visualization
def plot_barplot(df_plot, x_name, y_name, title):
    """
    Display stacked column chart of weights for criteria for `x_name == Weighting methods`
    and column chart of ranks for alternatives `x_name == Alternatives`

    Parameters
    ----------
        df_plot : dataframe
            dataframe with criteria weights calculated different weighting methods
            or with alternaives rankings for different weighting methods
        x_name : str
            name of x axis, Alternatives or Weighting methods
        y_name : str
            name of y axis, Ranks or Weight values
        title : str
            name of chart title, Weighting methods or Criteria
    """
    list_rank = np.arange(1, len(df_plot) + 1, 1)
    stacked = True
    width = 0.5
    if x_name == 'Alternatives':
        stacked = False
        width = 0.8
    else:
        df_plot = df_plot.T
    ax = df_plot.plot(kind='bar', width = width, stacked=stacked, edgecolor = 'black', figsize = (9,4))
    ax.set_xlabel(x_name, fontsize = 12)
    ax.set_ylabel(y_name, fontsize = 12)

    if x_name == 'Alternatives':
        ax.set_yticks(list_rank)

    ax.set_xticklabels(df_plot.index, rotation = 'horizontal')
    ax.tick_params(axis = 'both', labelsize = 12)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol=5, mode="expand", borderaxespad=0., edgecolor = 'black', title = title, fontsize = 11)

    ax.grid(True, linestyle = '--')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.show()


def draw_heatmap(data, title):
    """
    Display heatmap with correlations of compared rankings generated using different methods

    Parameters
    ----------
    data : dataframe
        dataframe with correlation values between compared rankings
    title : str
        title of chart containing name of used correlation coefficient
    """
    plt.figure(figsize = (6, 4))
    sns.set(font_scale=0.8)
    heatmap = sns.heatmap(data, annot=True, fmt=".2f", cmap="YlGn",
                          linewidth=0.5, linecolor='w')
    plt.yticks(va="center")
    plt.xlabel('Weighting methods')
    plt.title('Correlation coefficient: ' + title)
    plt.tight_layout()
    plt.show()


def plot_boxplot(data):
    """
    Display boxplot showing distribution of criteria weights determined with different methods.

    Parameters
    ----------
    data : dataframe
        dataframe with correlation values between compared rankings
    """
    
    plt.figure(figsize = (7, 4))
    
    ax = data.boxplot()
    ax.grid(True, linestyle = '--')
    ax.set_axisbelow(True)
    ax.set_xlabel('Alternatives', fontsize = 12)
    ax.set_ylabel('TOPSIS preference distribution', fontsize = 12)
    plt.tight_layout()
    plt.show()

# Create dictionary class
class Create_dictionary(dict):
  
    # __init__ function
    def __init__(self):
        self = dict()
          
    # Function to add key:value
    def add(self, key, value):
        self[key] = value


def main():
    filename = 'dataset_mobile_phones.csv'
    data = pd.read_csv(filename, index_col = 'Ai')
    # df_data = data.iloc[:len(data) - 2, :]
    df_data = data.iloc[:len(data) - 12, :]
    types = data.iloc[len(data) - 2, :].to_numpy()

    matrix = df_data.to_numpy()

    list_alt_names = [r'$A_{' + str(i) + '}$' for i in range(1, df_data.shape[0] + 1)]
    cols = [r'$C_{' + str(j) + '}$' for j in range(1, data.shape[1] + 1)]

    # part 1 - study with single distance metric

    # Determine criteria weights with chosen weighting method
    weights = mcda_weights.critic_weighting(matrix)

    # Create the TOPSIS method object
    topsis = TOPSIS(normalization_method = norms.minmax_normalization, distance_metric = dists.euclidean)

    # Calculate alternatives preference function values with TOPSIS method
    pref = topsis(matrix, weights, types)

    # rank alternatives according to preference values
    rank = rank_preferences(pref, reverse = True)

    # save results in dataframe
    df_results = pd.DataFrame(index = list_alt_names)
    df_results['Pref'] = pref
    df_results['Rank'] = rank

    
    # part 2 - study with several distance metrics
    # Create a list with distance metrics that you want to explore
    distance_metrics = [
        dists.euclidean,
        dists.manhattan,
        # dists.hausdorff,
        # dists.correlation,
        # dists.chebyshev,
        # dists.cosine,
        # dists.squared_euclidean,
        dists.bray_curtis,
        dists.canberra,
        dists.lorentzian,
        # dists.jaccard,
        # dists.dice,
        dists.hellinger,
        dists.matusita,
        dists.squared_chord,
        dists.pearson_chi_square,
        dists.squared_chi_square
    ]
    
    # Create dataframes for preference function values and rankings determined using distance metrics
    df_preferences = pd.DataFrame(index = list_alt_names)
    df_rankings = pd.DataFrame(index = list_alt_names)

    for distance_metric in distance_metrics:
        # Create the TOPSIS method object
        topsis = TOPSIS(normalization_method = norms.minmax_normalization, distance_metric = distance_metric)
        pref = topsis(matrix, weights, types)
        rank = rank_preferences(pref, reverse = True)
        df_preferences[distance_metric.__name__.capitalize().replace('_', ' ')] = pref
        df_rankings[distance_metric.__name__.capitalize().replace('_', ' ')] = rank
        

    # plot box chart of alternatives preference values
    plot_boxplot(df_preferences.T)

    # plot column chart of alternatives rankings
    plot_barplot(df_rankings, 'Alternatives', 'Rank', 'Distance metric')

    
    # Plot heatmaps of rankings correlation coefficient
    # Create dataframe with rankings correlation values
    results = copy.deepcopy(df_rankings)
    method_types = list(results.columns)
    dict_new_heatmap_p = Create_dictionary()

    for el in method_types:
        dict_new_heatmap_p.add(el, [])

    for i, j in [(i, j) for i in method_types[::-1] for j in method_types]:
        dict_new_heatmap_p[j].append(corrs.pearson_coeff(results[i], results[j]))
            
    df_new_heatmap_p = pd.DataFrame(dict_new_heatmap_p, index = method_types[::-1])
    df_new_heatmap_p.columns = method_types

    # Plot heatmap with rankings correlation
    draw_heatmap(df_new_heatmap_p, r'$Pearson$')
    

if __name__ == '__main__':
    main()