from mlp.multi_label_propagation import *
from soil_microbiome.modeling import *

# taxonomic level is species
tax_level = 'species'

# we choose all input variables as environmental variables
env_vars = global_vars['input_variables']

"""
k = 40 # number of species of interest
species = Species(file_name="species.xlsx", x_dim=len(all_input_variables), env_vars=env_vars, tax_level=tax_level)
data_split_train_test(species)
"""

# print species info
#species.print_info()


num_species = 20

"""
species_train = Species(file_name="species_train.xlsx", x_dim=len(global_vars['input_variables']), \
                        env_vars=env_vars, tax_level=tax_level, num_species_interest=num_species)

MLTrain(species_train).train()


"""

species_test = Species(file_name="species_test.xlsx", x_dim=len(global_vars['input_variables']), env_vars=env_vars, tax_level=tax_level, num_species_interest=num_species)

MLEvaluate(species_test).evaluate()

#MLTrain(species).train()

#MLEvaluate(species).evaluate()

#MLEvaluate(species).hf_predict()
