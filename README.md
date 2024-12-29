## AMF Prediction

This project is dedicated to the analysis and prediction of arbuscular mycorrhizal fungi (AMF) data obtained from [Global AM Fungi](https://globalamfungi.com/) website. 

See the [GlobalAMF.ipynb] for detailed analysis (still ongoing) of the data. 

Predictive models used are:

1. Nearest-neighbour based approaches: Harmonic Function, (multi-label) k-nearest neighbour
2. Tree-based methods: Random Forest, Gradient Boosting, LightGBM, XGBoost, CatBoost
3. Margin classifier: Support-vector machines
4. Multi-label learners: Ensembles of classifier chains (based on random forest), Label Powerset (based on support vector machine)
5. Auto ML, ensembles of methods/stacking: Autogluon

All methods, except ecc and lp, are binary relevance learners: learning/classification is done for each label separately.
On the other hand, ecc and lp leverage label relationships.



### Running instructions
```
git clone https://github.com/kmusayeva/GlobalAMFPred
```

To train the models:

```
python main.py --mode train --num_species 20
```

To test the models:
```
python main.py --mode eval --num_species 20
```


### Licence
This project is licensed under the MIT Licence - see the LICENCE.md file for details.


### Author
* Khadija Musayeva, PhD 
* Email: [khmusayeva@gmail.com](khmusayeva@gmail.com)


