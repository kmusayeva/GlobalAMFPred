## AMF Prediction

This project addresses soil microbiome prediction problem in the context of multi-label classification. The focus here is specifically on arbuscular mycorrhizal fungi (AMF).
The AMF data used is publicly available on [Global AM Fungi](https://globalamfungi.com/) website. 

Predictive models used are:

1. Nearest-neighbour based approaches: Harmonic Function (hf), (multi-label) k-nearest neighbour
2. Random Forest (rf), Gradient Boosting (gb), LightGBM (lgbm), XGBoost, CatBoost
3. Ensembles of classifier chains (ecc), Label Powerset (lp)
4. Support-vector machines (svm)
5. Autogluon

Here, most methods, hf, rf, gb, k-nnn, svm, lgbm, xgboost and autogluon are binary relevance learners: learning is done for each label separately.
On the other hand, ecc and lp leverage the label relationship.



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


