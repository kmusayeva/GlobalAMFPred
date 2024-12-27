from mlp.multi_label_propagation import *
from soil_microbiome.modeling import *

# taxonomic level is species
tax_level = 'species'

# we choose all input variables as environmental variables
env_vars = global_vars['input_variables']


k = 20 # number of species of interest

"""
species = Species(file_name="species.xlsx", x_dim=len(global_vars['input_variables']), env_vars=env_vars, tax_level=tax_level, num_species_interest=k)
iterative_data_split(species)
"""

# print species info
# species.print_info()


"""
species_train = Species(file_name="species_train.xlsx", x_dim=len(global_vars['input_variables']), \
                        env_vars=env_vars, tax_level=tax_level, num_species_interest=k)
MLTrain(species_train).train()
"""


"""
species_test = Species(file_name="species_test.xlsx", x_dim=len(global_vars['input_variables']), \
                        env_vars=env_vars, tax_level=tax_level, num_species_interest=k)
MLEvaluate(species_test).evaluate()
"""

#MLEvaluate(species_test).autogluon_predict()

#MLTrain(species).train()

#MLEvaluate(species).evaluate()

#MLEvaluate(species).hf_predict()