import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from soil_microbiome.modeling.species import *


def test_iter_strat():

    num_species = 20
    species = Species(
        file_name="species.xlsx",
        x_dim=len(global_vars["input_variables"]),
        env_vars=global_vars["input_variables"],
        tax_level="species",
        num_species_interest=num_species,
    )

    assert (species.Y_top.sum(axis=1) == 0).sum() > 0


if __name__ == "__main__":
    test_iter_strat()
