from mlp.multi_label_propagation import *
from soil_microbiome.modeling import *
import argparse

# taxonomic level is species
tax_level = "species"

# we choose all input variables as environmental variables
env_vars = global_vars["input_variables"]


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate a species model.")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "eval"],
        help="Mode to run: 'train' or 'eval'.",
    )
    parser.add_argument(
        "--num_species", type=int, required=True, help="Number of top frequent species."
    )

    parser.add_argument(
        "--method",
        type=str,
        nargs='*',
        required=False,
        help="Name(s) of learning method(s)."
    )

    args = parser.parse_args()

    if args.mode == "train":
        species = Species(
            file_name="species_train.xlsx",
            x_dim=len(global_vars["input_variables"]),
            env_vars=env_vars,
            tax_level=tax_level,
            num_species_interest=args.num_species,
        )
        MLTrain(species, args.method).train()

    elif args.mode == "eval":
        species = Species(
            file_name="species_test.xlsx",
            x_dim=len(global_vars["input_variables"]),
            env_vars=env_vars,
            tax_level=tax_level,
            num_species_interest=args.num_species,
        )
        MLEvaluate(species, args.method).evaluate()

if __name__ == "__main__":
    main()
