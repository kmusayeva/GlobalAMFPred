## Soil Microbiome Prediction

This project addresses soil microbiome prediction problem in the context of multi-label classification. The focus here is specifically on arbuscular mycorrhizal fungi (AMF).
The AMF data used is publicly available on [Global AM Fungi](https://globalamfungi.com/). 

### Project directory
The `AMF-Preds` directory consists of two main subdirectories:`soil_microbiome` and `m_lp`. 

`soil_microbiome/data/GlobalAMFungi` contains the tabular data relating environmental variables with AMF abundancies. 
Currently, the provided data concern the taxonomic level of species. The data concerning each taxonomic level should be 
placed in the corresponding folder. For example, `soil_microbiome/data/GlobalAMFungi/Species` for the species level, 
`soil_microbiome/data/GlobalAMFungi/Genus` for genus level.

`soil_microbiome/data_analysis` contains code to evaluate off-the-shelf multi-label classification methods, as well as label-propagation 
approaches implemented in `m_lp` project. 

The project `m_lp` implements label-propagation approaches (please check [2] for more information).


### Evaluation strategy
The performance is evaluated based on multiple metrics: Hamming loss, subset accuracy, and the family of F1 measures.
Due to the "power-law" like distribution of the species abundancies (at the taxonomic level of species), i.e., few very abundant, 
mostly rare, only top frequent species should be selected (an option to be provided by a user). In such a setting, 
the family of F1-measures provides the most accurate picture of predictive performance. 
Furthermore, the label distribution should be kept similar across the folds for the same reason. 
In this project, this is done based on the stratified sampling method of Sechidis et al. [1].


### Running instructions
```
git clone https://github.com/kmusayeva/AMF-Preds
```
You will need to install `mlp` and `soil_microbiome` projects separately as follows:
```
cd AMF-Preds/mlp
pip install .
```

```
cd AMF-Preds/soil_microbiome
pip install .
```

The entry point of the project is `AMF-Preds/main.py`:

```commandline
cd AMF-preds
python main.py
```

You can modify `main.py` to:
* specify environmental variables, for instance, ```env_vars = ['pH', 'MAP', 'MAT']```
* specify taxonomic level at which to do prediction ```tax_level = 'Species'```
* choose n number of top frequent species ```species.get_top_species(num_top=10)```


### Licence
This project is licensed under the MIT Licence - see the LICENCE.md file for details.


### Author
* Khadija Musayeva, PhD 
* Email: [khmusayeva@gmail.com](khmusayeva@gmail.com)

### Version history
* 0.1 Initial Release 


### References
1. Sechidis, K., Tsoumakas, G., & Vlahavas, I. (2011). On the stratification of multi-label data. 
2. Musayeva, K., & Binois, M. (2023). Improved Multi-label Propagation for Small Data with Multi-objective Optimization. 
