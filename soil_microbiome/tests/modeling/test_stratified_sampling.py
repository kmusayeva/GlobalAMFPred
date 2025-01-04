import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from soil_microbiome.modeling.stratified_sampling import *

def test_iterative_stratification():
    # Setup any required inputs for the function
    Y = np.array(
        [
            [1, 1, 1, 0, 1, 1, 0, 1, 1, 1.0],
            [0, 0, 1, 0, 1, 1, 1, 0, 0, 1.0],
            [0, 0, 1, 0, 1, 1, 1, 0, 0, 1.0],
            [0, 0, 1, 0, 1, 1, 1, 0, 0, 1.0],
            [1, 0, 0, 0, 0, 0, 1, 1, 1, 1.0],
            [0, 0, 1, 0, 0, 1, 1, 1, 0, 1.0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1.0],
            [0, 1, 0, 0, 1, 1, 0, 0, 1, 1.0],
            [0, 0, 0, 1, 0, 0, 1, 1, 1, 1.0],
            [0, 0, 0, 1, 0, 0, 1, 1, 0, 0.0],
            [1, 0, 0, 0, 0, 0, 1, 1, 1, 0.0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 1.0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0.0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0.0],
            [0, 1, 0, 0, 1, 1, 0, 0, 0, 1.0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1.0],
            [0, 0, 0, 1, 0, 0, 1, 1, 1, 1.0],
            [1, 0, 0, 1, 1, 0, 1, 1, 1, 1.0],
            [0, 0, 0, 1, 1, 0, 0, 1, 1, 1.0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0],
            [0, 1, 0, 0, 1, 1, 0, 1, 1, 1.0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0.0],
            [1, 0, 1, 1, 0, 0, 0, 1, 1, 1.0],
            [0, 0, 0, 1, 0, 0, 1, 0, 1, 0.0],
            [0, 0, 0, 1, 0, 0, 1, 1, 0, 0.0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1.0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1.0],
            [0, 0, 0, 1, 0, 0, 0, 1, 1, 1.0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0.0],
        ]
    )
    k = 2
    proportions = [0.8, 0.2]
    train_indices, test_indices = iterative_stratification(Y, k, proportions)
    result = any(element in train_indices for element in test_indices)
    expected_result = False
    assert result == expected_result, "There are common indices in train and test sets."


