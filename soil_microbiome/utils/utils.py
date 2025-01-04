import os
import numpy as np
import pandas as pd
from soil_microbiome import global_vars
from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.model_selection import IterativeStratification
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

    file = os.path.join(global_vars["data_dir"], filename)

    return file


def read_file(filename):
    name, ext = os.path.splitext(filename)

    if ext not in [".xlsx", ".csv"]:
        raise ValueError("Please specify either xlsx or csv file.")

    full_path = os.path.join(global_vars["data_dir"], filename)

    if not os.path.isfile(full_path):
        raise ValueError("File does not exist.")

    if ext == ".xlsx":
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
        print(f"{func.__name__} : {total_time:.4f} seconds")
        return result

    return timeit_wrapper


def haversine(coord1, coord2):
    return great_circle(coord1, coord2).km


def iterative_data_split(species):

    # create three subsets of data for training, validation and testing, with proportions 60%/20%/20%.

    #Y = species.Y_top[(species.Y_top.sum(axis=1) != 0)] # remove all observations with no top selected species
    #X = species.X[(species.Y_top.sum(axis=1) != 0)]

    num_species = species.Y_top.shape[1]

    if num_species < 10:
        order = num_species
    else:
        order = 20

    stratifier = IterativeStratification(
        n_splits=2, order=order, sample_distribution_per_fold=[0.2, 0.8]
    )

    train_valid_idx, test_idx = next(stratifier.split(species.X, species.Y_top))

    X_train_valid, Y_train_valid = (
        species.X.iloc[train_valid_idx],
        species.Y_top.iloc[train_valid_idx],
    )

    Y_test = species.Y_top.iloc[test_idx]
    X_test = species.X.iloc[test_idx]

    df_train = pd.DataFrame(X_train_valid, columns=species.X.columns).join(
        pd.DataFrame(Y_train_valid, columns=species.Y_top.columns)
    )
    df_test = pd.DataFrame(X_test, columns=species.X.columns).join(
        pd.DataFrame(Y_test, columns=species.Y_top.columns)
    )

    df_train.to_excel(
        os.path.join(global_vars["data_dir"], "species_train.xlsx"), index=False
    )
    df_test.to_excel(
        os.path.join(global_vars["data_dir"], "species_test.xlsx"), index=False
    )
