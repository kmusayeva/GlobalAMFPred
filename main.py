from mlp.multi_label_propagation import *
from soil_microbiome.data_analysis import *
from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.model_selection import IterativeStratification
# taxonomic level is species
tax_level = 'species'

# prepare environmental variables
all_input_variables = ['anthropogenic', 'aquatic', 'cropland', 'desert', 'forest', 'grassland',
        'shrubland', 'tundra', 'wetland', 'woodland', 
       'longitude', 'latitude','MAP', 'MAT', 'pH']

# we choose all input variables as environmental variables
env_vars = all_input_variables

species = Species(tax_level=tax_level, x_dim=len(all_input_variables), env_vars=env_vars)

# get k top frequent species
k = 20
species.get_top_species(k)


"""
stratifier = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[0.2, 0.8])

train_valid_idx, test_idx = next(stratifier.split(species.X.to_numpy(), species.Y.to_numpy()))

X_train_valid, Y_train_valid = species.X.to_numpy()[train_valid_idx], species.Y.to_numpy()[train_valid_idx]

stratifier = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[0.25, 0.75])

train_idx, valid_idx = next(stratifier.split(X_train_valid, Y_train_valid))

X_dist_squared = dist_matrix(StandardScaler().fit_transform(X_train_valid)) ** 2

print(X_dist_squared[valid_idx])


X_dist_squared = dist_matrix(StandardScaler().fit_transform(species.X.to_numpy())) ** 2

print(X_dist_squared[test_idx])
"""

# print species info
species.print_info()

model = MLClassification(species)
model.hf_predict()
