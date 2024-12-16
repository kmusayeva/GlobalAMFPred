"""
Creates species from the species data set (this path should be set in the global_vars)
and the taxonomic level: order, family, gender or species.
Author: Khadija Musayeva
Email: khmusayeva@gmail.com
"""
import os.path
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from soil_microbiome.utils.utils import *
import seaborn as sns
from typing import List, Tuple, Dict, Optional


class Species:
    def __init__(self, tax_level: str, x_dim: int, env_vars: Optional[List[str]] = None,
                 species_pattern: Optional[List[str]] = None,
                 species: Optional[List[str]] = None, is_global_amf: Optional[bool] = False) -> None:
        """
        Initialize the Species object with X as environmental variables, Y abundancy matrix,
        Yb absence/presence matrix, Y_top top most frequent species. Calculate label ditrbution,
        class imbalance information.
        @param tax_level: taxonomic level: order, family, gender, or species
        @param x_dim: total number of variables in input data
        @param env_vars: list of selected environmental/input variables
        @param species_pattern: select species based on their names, such as "glo"
        @param species: list of species such as ["Glomus", "Rhizophagus"]
        @param is_global_amf: if true, reads from GlobalAMFungi/Species directory, otherwise from data/tax_level directory
        """
        self.level = tax_level
        self.env_vars = env_vars
        self.x_dim = x_dim
        self.is_binary = False
        self.X, self.Y = self.__read_data(is_global_amf)
        self.Y = self.Y.loc[:, (self.Y != 0).any(axis=0)]
        if species is not None:
            self.Y = self.Y[species]
        if species_pattern is not None:
            pattern = '|'.join(species_pattern)
            self.Y = self.Y.loc[:, self.Y.columns.str.contains(pattern)]
        self.Yb = self.Y.where(self.Y == 0, 1)
        self.freq = ((self.Yb.sum().sort_values() * 100 / len(self.Y)).round(2))[::-1]
        self.Yb_sorted = self.Yb.sort_index(axis=1)
        self.Y_top = self.Yb
        self.label_distri = self.label_info()

    def __read_data(self, is_global_amf: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Read data from the specified file and return environmental variables and species data.
        @param is_global_amf: read from global amf data?
        @return: input, output matrices
        """
        data_dir = 'global_amf_dir' if is_global_amf else 'data_dir'
        file_name = os.path.join(global_vars[data_dir], f'{self.level}/{self.level}.xlsx')
        try:
            data = read_file(file_name)
        except ValueError as err:
            print(f"Error in reading file {file_name}: {err}")
            sys.exit(1)
        if self.env_vars is None:
            self.env_vars = data.columns.tolist()[:self.x_dim]
        return data[self.env_vars], data.iloc[:, self.x_dim:]

    def label_info(self) -> Dict[str, float]:
        """Calculate label distribution, density, and average class imbalance."""
        n, c = self.Y_top.shape
        col_sum = np.sum(self.Y_top.to_numpy(), axis=0)
        row_sum = np.sum(self.Y_top.to_numpy())
        imbalance_ratios = np.maximum(col_sum, n - col_sum) / np.minimum(col_sum, n - col_sum)
        unique_rows, counts = np.unique(self.Y_top, axis=0, return_counts=True)
        info = {'Number of unique labelsets (ls)': len(unique_rows),
                'Label density': (row_sum / (n * c)).round(2),
                'Class imbalance': np.mean(imbalance_ratios).round(2),
                'Max example per ls': counts.max(),
                'Mean example per ls': counts.mean().round(0),
                'Min example per ls': counts.min()
                }
        return info

    def standardize_env_vars(self, v: Optional[List[str]] = None) -> None:
        """Standardize environmental variables."""
        if v is None:
            self.X = StandardScaler().fit_transform(self.X)
        else:
            self.X[v] = StandardScaler().fit_transform(self.X[v])

    def min_max_norm_env_vars(self, v: Optional[List[str]] = None) -> None:
        """
        Normalize environmental variables using MinMaxScaler.
        @param v: list of environmental variables
        """
        if v is None:
            self.X = MinMaxScaler().fit_transform(self.X)
        else:
            self.X[v] = MinMaxScaler().fit_transform(self.X[v])

    def get_top_species(self, num_top: int, exclude_first_n: Optional[int] = None) -> None:
        """ 
        Select top most frequent species. Update label distribution information for these species.
        @param num_top: number of top frequent species
        @param exclude_first_n: to exclude highly frequent species
        """
        if exclude_first_n is None:
            exclude_first_n = 0
        if num_top > self.Y.shape[1] or exclude_first_n > num_top:
            raise ValueError('Start and end values are not correct.')
        self.freq = self.freq[exclude_first_n:num_top]
        cols = self.freq.index.tolist()
        self.Y_top = self.Yb[cols]
        self.label_distri = self.label_info()

    def plot_freq(self, save_plot: Optional[bool] = False, name: Optional[str] = None) -> None:
        """
        Plots and saves barplots of species frequencies.
        @param name: name of the file to save to
        @param save_plot: show plot
        """
        self.freq.plot.barh(x='Relative frequency', y="Species", rot=0)
        plt.tight_layout()
        if save_plot:
            if not name:
                raise ValueError('Please specify the name of the figure.')
            plt.savefig(f"../figures/{name}.png", dpi=300, bbox_inches='tight')
        plt.show()

    def plot_hist_env_vars_species(self, param: str, eps: float) -> None:
        """
        For each species, plots the histogram of a given environmental parameter
        and saves to the file.
        @param param: the name of the environmental parameter
        @param eps: bin width
        """
        pr1 = self.X[param]
        for sp in self.Y_top.columns.tolist():
            ind = self.Y_top[sp] > 0
            pr2 = self.X[ind][param]
            plt.hist(pr1, bins=np.arange(min(pr1), max(pr1) + eps, eps), alpha=0.5, label='All ' + param)
            plt.hist(pr2, bins=np.arange(min(pr1), max(pr1) + eps, eps), alpha=0.5, label=sp)
            plt.legend(loc='upper left')
            plt.xlabel(param)
            plt.savefig('../figures/' + param + ' ' + sp)
            plt.clf()

    def plot_hist_env_vars(self, param: str, eps: float, prefix: str) -> None:
        """
        Plots histogram of a given environmental parameter
        @param param: the name of the environmental parameter
        @param eps: bin width
        @param prefix: used in the name of the plot
        """
        pr1 = self.X[param]
        plt.hist(pr1, bins=np.arange(min(pr1), max(pr1) + eps, eps), alpha=0.5, label=prefix)
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel(param)
        plt.savefig(f'../figures/{prefix}{param}.png')

    def plot_ph_heatmap(self, plot_name: str) -> None:
        """
        Groups pH into the groups of acidity.
        PLots the distribution of species in each group of acidity.
        @param plot_name: the name of the plot
        """
        bins = [0, 4.5, 5.5, 6.5, 7.5, 8.5, float('inf')]  # float('inf') covers all pH greater than 8.5
        labels = ['<4.5', '4.5-5.5', '5.5-6.5', '6.5-7.5', '7.5-8.5', '>8.5']
        # labels = ['very acidic', 'a little acidic', 'acidic', 'neutral', 'basic', 'very basic']
        ph_levels = pd.cut(self.X["pH"], bins=bins, labels=labels, include_lowest=True)
        # (self.Yb.groupby(ph_levels).sum()).to_excel(os.path.join(global_vars["data_dir"], "richness.xlsx"))
        # Setting the aesthetics for the heatmap
        richness = (self.Y_top.groupby(ph_levels).sum()).astype(int)
        richness["pH"] = richness.index
        # richness.set_index('pH')
        plt.figure(figsize=(15, 10))
        ax = sns.heatmap(richness.set_index('pH'), cmap='viridis', annot=True, fmt="d")
        # plt.title('Heatmap of Species Count Across Different pH Levels', fontsize=20)
        plt.xlabel('Species', fontsize=14)
        plt.ylabel('pH Levels', fontsize=14)
        plt.xticks(rotation=45, fontsize=12, ha='right')
        plt.yticks(rotation=0, fontsize=12)
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig('../figures/' + plot_name + '.png')
        # plt.show()
        # print(self.Yb.groupby(ph_levels).sum())
        print(self.X["pH"].groupby(ph_levels).count())

    def print_info(self) -> None:
        """
        Prints out label distribution.
        """
        print(f">>>Frequencies of top species: \n{self.freq.to_string()}")
        print(f">>>The number of examples is {self.X.shape[0]} and the number of labels is {self.Y.shape[1]}.")
        print('\n'.join(f"{key}: {value}" for key, value in self.label_distri.items()))
