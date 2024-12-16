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
