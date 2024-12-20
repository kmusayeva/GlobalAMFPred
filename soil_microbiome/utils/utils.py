import os
import numpy as np
import pandas as pd
from soil_microbiome import global_vars
import matplotlib.pyplot as plt
from functools import wraps
import time
import random
from geopy.distance import great_circle


def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully.")

    else:
        print(f"Directory '{directory}' already exists.")


def check_extension(filename, extension):
    # Convert the filename to lowercase to make the comparison case-insensitive
    lowercase_filename = filename.lower()

    # Check if the filename already has the extension
    if not lowercase_filename.endswith(extension):
        # Add the extension to the filename
        filename += extension

    file = os.path.join(global_vars['data_dir'], filename)

    return file


def read_file(filename):
    name, ext = os.path.splitext(filename)

    if ext not in ['.xlsx', '.csv']:
        raise ValueError('Please specify either xlsx or csv file.')

    full_path = os.path.join(global_vars['data_dir'], filename)

    if not os.path.isfile(full_path):
        raise ValueError('File does not exist.')

    if ext == '.xlsx':
        data = pd.read_excel(full_path)

    else:
        data = pd.read_csv(full_path)

    return data


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'{func.__name__} : {total_time:.4f} seconds')
        return result

    return timeit_wrapper


def haversine(coord1, coord2):
    return great_circle(coord1, coord2).km




def data_split_train_test(species):

    ### to be updated

    all_input_variables = ['anthropogenic', 'aquatic', 'cropland', 'desert', 'forest', 'grassland',
        'shrubland', 'tundra', 'wetland', 'woodland', 
        'longitude', 'latitude','MAP', 'MAT', 'pH']

    # we choose all input variables as environmental variables
    env_vars = all_input_variables

    species = Species(tax_level=tax_level, x_dim=len(all_input_variables), env_vars=env_vars)

    # get k top frequent species
    k = 40
    species.get_top_species(k)

    # create three subsets of data for training, validation and testing, with proportions 60%/20%/20%.

    stratifier = IterativeStratification(n_splits=2, order=5, sample_distribution_per_fold=[0.2, 0.8])

    train_valid_idx, test_idx = next(stratifier.split(species.X, species.Y_top))

    X_train_valid, Y_train_valid = species.X.iloc[train_valid_idx], species.Y_top.iloc[train_valid_idx]

    Y_test = species.Y_top.iloc[test_idx]
    X_test = species.X.iloc[test_idx]

    #print(Y_train_valid.sum(axis=0).sort_values(ascending=False))
    #print(Y_test.sum(axis=0).sort_values(ascending=False))

    #df_train = np.concatenate((X_train_valid, Y_train_valid), axis=1)
    #df_test = np.concatenate((X_test, Y_test), axis=1)

    df_train = pd.DataFrame(X_train_valid, columns=species.X.columns).join(pd.DataFrame(Y_train_valid, columns=species.Y_top.columns))
    df_test = pd.DataFrame(X_train_valid, columns=species.X.columns).join(pd.DataFrame(Y_train_valid, columns=species.Y_top.columns))

    #print(df_train.iloc[:, (len(env_vars)):].sum(axis=0).sort_values(ascending=False))

    df_train.to_excel(os.path.join(global_vars["data_dir"], "species_train.xlsx"), index=False)
    df_test.to_excel(os.path.join(global_vars["data_dir"], "species_test.xlsx"), index=False)