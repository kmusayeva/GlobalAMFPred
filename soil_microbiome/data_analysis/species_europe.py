"""
Global AMF species containing observations only from Europe continent.
Author: Khadija Musayeva
Email: khmusayeva@gmail.com
"""
from .species import *


class SpeciesEurope(Species):

    def __init__(self, tax_level: str, x_dim: int,
                 env_vars: Optional[List[str]] = None,
                 species_pattern: Optional[List[str]] = None,
                 species: Optional[List[str]] = None, is_global_amf=True) -> None:
        """
        Initialize Europe species from global amf data.
        @param tax_level: taxonomic level: order, family, gender, or species
        @param x_dim: total number of variables in input data
        @param env_vars: list of selected environmental/input variables
        @param species_pattern: select species based on their names, such as "glo"
        @param species: list of species such as ["Glomus", "Rhizophagus"]
        @param is_global_amf: reads from GlobalAMFungi/Species directory
        """
        super().__init__(tax_level, x_dim, env_vars, species_pattern, species, is_global_amf=True)
        self.X, self.Y = self.__read_data()
        self.Y = self.Y.loc[:, (self.Y != 0).any(axis=0)]
        if species:
            self.Y = self.Y[species]
        if species_pattern:
            self.Y = self.Y.loc[:, self.Y.columns.str.contains('|'.join(species_pattern))]
        self.Yb = self.Y.where(self.Y == 0, 1, inplace=False)
        self.freq = ((self.Yb.sum().sort_values() * 100 / len(self.Y)).round(2))[::-1]
        self.Yb_sorted = self.Yb.sort_index(axis=1)
        self.Y_top = self.Yb
        self.label_distri = self.label_info()

    def __read_data(self, is_global_amf=True):
        file_name = os.path.join(global_vars['global_amf_dir'], f'{self.level}/{self.level}.xlsx')
        try:
            data = read_file(file_name)
        except ValueError as err:
            print("Error in reading the species file.")
            sys.exit(1)
        data = data[data['continent'] == "Europe"]
        if self.env_vars is None:
            self.env_vars = data.columns.tolist()[:self.x_dim]
        return data[self.env_vars], data.iloc[:, self.x_dim:]
