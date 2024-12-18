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


stratifier = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[0.2, 0.8])
train_valid_indices, test_indices = next(stratifier.split(species.X, species.Y_top))
stratifier = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[0.2, 0.8])
train_indices, valid_indices = next(stratifier.split(species.X.iloc[train_valid_indices], species.Y_top.iloc[train_valid_indices]))


print({int(x) for x in train_indices}<={int(x) for x in train_valid_indices})


print(train_valid_indices)
print(train_indices)

# print species info
#species.print_info()

#model = MLClassification(species)
#model.hf_predict()

#train_indices, valid_indices = next(stratifier.split(model.species.X[train_valid_indices], model.species.Y[train_valid_indices]))

#print(valid_indices)
#print(test_indices)


# perform multi-label classification
#model = MLClassification(species)
#model.evaluate()
#model.printResults()
