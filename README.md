## AMF Prediction

This project is dedicated to the analysis and prediction of arbuscular mycorrhizal fungi (AMF) data obtained from [Global AM Fungi](https://globalamfungi.com/) website. 

See the [GlobalAMF.ipynb] for preliminary analysis (still ongoing) of the data. 

Predictive models used are:

1. Nearest-neighbour based approaches: Harmonic Function (hf), (multi-label) k-nearest neighbour
2. Tree-based methods: Random Forest (rf), Gradient Boosting (gb), LightGBM (lgbm), XGBoost, CatBoost
3. Multi-label learners: Ensembles of classifier chains (ecc), Label Powerset (lp)
4. Margin classifier: Support-vector machines (svm)
5. Ensembles of many methods: Autogluon

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


