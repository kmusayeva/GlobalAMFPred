import os

from pathlib import Path

base_dir = Path(__file__).parent

global_vars = {

'data_dir' : base_dir/'data',

'model_dir' : base_dir/'modeling'/'models',

'input_variables' : ['anthropogenic', 'aquatic', 'cropland', 'desert', 'forest', 'grassland',
        'shrubland', 'tundra', 'wetland', 'woodland', 
       'longitude', 'latitude','MAP', 'MAT', 'pH']


    }
