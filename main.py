from mlp.multi_label_propagation import *
from soil_microbiome.data_analysis import *

# taxonomic level is species
tax_level = 'Species'

# prepare environmental variables
soil_usage = ['anthropogenic', 'cropland', 'desert', 'forest', 'grassland', 'shrubland', 'tundra', 'woodland']
soil_vars = ['pH', 'MAP', 'MAT']
env_vars = [*soil_usage, *soil_vars]

# create europe species object
species = SpeciesEurope(tax_level=tax_level, x_dim=17, env_vars=env_vars, is_global_amf=True)

# get 30 top frequent species
species.get_top_species(10)

# print species info
species.print_info()

# perform multi-label classification
model = MLClassification(species)
model.evaluate(nshuffle=2)
model.printResults()
