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

species = Species(tax_level=tax_level, x_dim=len(all_input_variables), env_vars=env_vars)

# get k top frequent species
k = 40
species.get_top_species(k)


# print species info
#species.print_info()


#print(global_vars["model_dir"])

#MLTrain(species).train()

#MLEvaluate(species).evaluate()

#MLEvaluate(species).hf_predict()

stratifier = IterativeStratification(n_splits=2, order=5, sample_distribution_per_fold=[0.2, 0.8])

train_valid_idx, test_idx = next(stratifier.split(species.X, species.Y_top))

X_train_valid, Y_train_valid = species.X.iloc[train_valid_idx], species.Y_top.iloc[train_valid_idx]

Y_test = species.Y_top.iloc[test_idx]
X_test = species.X.iloc[test_idx]

print(Y_train_valid.sum(axis=0).sort_values(ascending=False))
print(Y_test.sum(axis=0).sort_values(ascending=False))

#df_train = np.concatenate((X_train_valid, Y_train_valid), axis=1)
#df_test = np.concatenate((X_test, Y_test), axis=1)

df_train = pd.DataFrame(X_train_valid, columns=species.X.columns).join(pd.DataFrame(Y_train_valid, columns=species.Y_top.columns))
df_test = pd.DataFrame(X_train_valid, columns=species.X.columns).join(pd.DataFrame(Y_train_valid, columns=species.Y_top.columns))

#print(df_train.iloc[:, (len(env_vars)):].sum(axis=0).sort_values(ascending=False))


df_train.to_excel(os.path.join(global_vars["data_dir"], "species_train.xlsx"), index=False)
df_test.to_excel(os.path.join(global_vars["data_dir"], "species_test.xlsx"), index=False)