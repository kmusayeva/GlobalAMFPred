from mlp.multi_label_propagation import *
from soil_microbiome.modeling import *
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

species = Species(file_name="species_train.xlsx", x_dim=len(all_input_variables), tax_level=tax_level, env_vars=env_vars)

# get k top frequent species
k = 20

species.get_top_species(k)

# print species info
species.print_info()

ml = MLClassification(species)



#print(global_vars["model_dir"])

#MLTrain(species).train()

#MLEvaluate(species).evaluate()

#MLEvaluate(species).hf_predict()

