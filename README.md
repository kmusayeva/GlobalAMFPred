## AMF Prediction

This project is dedicated to the analysis and prediction of arbuscular mycorrhizal fungi (AMF) data obtained from [Global AM Fungi](https://globalamfungi.com/) website. 

See the [GlobalAMF.ipynb] for detailed analysis (still ongoing). 

Predictive models used are:

1. Nearest-neighbour based approaches: (multi-label) k-nearest neighbour (knn)
2. Label propagation: harmonic function (hf)
3. Tree-based methods: random forest (rf), gradient boosting (gb), LightGBM (lgbm), XGBoost (xgb), CatBoost
4. Large-margin classifier: support-vector machine (svm)
5. Multi-label learners: ensembles of classifier chains (ecc), label powerset (lp)
6. Auto ML, ensembles of methods/stacking: autogluon 

All methods, except ecc and lp, are binary relevance learners: learning/classification is done for each label separately.
On the other hand, ecc and lp leverage label relationships. They both are based on random forest classifier.

Harmonic function is a transductive learning method.


### Running instructions
```
git clone https://github.com/kmusayeva/GlobalAMFPred
```

#### Training:

To train all models:

```
# num_species is the number of top most frequent species

python main.py --mode train --num_species 20
```

To train some models:

```
python main.py --mode train --num_species 20 --method knn
```
will train only knn, or

```
python main.py --mode train --num_species 20 --method  rf autogluon
```
will train rf and autogluon.

#### Evaluation:

To evaluate all models:

```
# for evaluation, num_species should not be larger than the one the models are trained for

python main.py --mode eval --num_species 20
```

To evaluate some models:

```
# for evaluation, num_species should not be larger than the one the models are trained for

python main.py --mode eval --num_species 20 --method ecc xgb
```
will evaluate only ecc and xgb.


### Licence
This project is licensed under the MIT Licence - see the LICENCE.md file for details.


### Author
* Khadija Musayeva, PhD 
* Email: [khmusayeva@gmail.com](khmusayeva@gmail.com)


