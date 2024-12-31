## AMF Prediction

This project is dedicated to the analysis and prediction of arbuscular mycorrhizal fungi (AMF) data obtained from [Global AM Fungi](https://globalamfungi.com/) website. 

See the [GlobalAMF.ipynb] for detailed analysis (still ongoing). 

Predictive models used are:

1. Nearest-neighbour based approaches: (multi-label) k-nearest neighbour
2. Label propagation: harmonic function
3. Tree-based methods: random forest, gradient boosting, LightGBM, XGBoost, CatBoost
4. Large-margin classifier: support-vector machines
5. Multi-label learners: ensembles of classifier chains (ecc, based on random forest), label powerset (lp, based on support vector machine)
6. Auto ML, ensembles of methods/stacking: Autogluon

All methods, except ecc and lp, are binary relevance learners: learning/classification is done for each label separately.
On the other hand, ecc and lp leverage label relationships.

Harmonic function is a transductive learning method.


### Running instructions
```
git clone https://github.com/kmusayeva/GlobalAMFPred
```

To train the models:

```
# num_species is the number of top most frequent species

python main.py --mode train --num_species 20
```

To test the models:
```
# for evaluation, num_species should not be larger than the one the models are trained for

python main.py --mode eval --num_species 20
```


### Licence
This project is licensed under the MIT Licence - see the LICENCE.md file for details.


### Author
* Khadija Musayeva, PhD 
* Email: [khmusayeva@gmail.com](khmusayeva@gmail.com)


