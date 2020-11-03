# Line of work
The link of the kaggle is here https://www.kaggle.com/t/4795897c22624e94a586eefa161fa61f  
All that is written here can be changed, redacted, removed, and so on, this is just a basis to have something to discuss.


### Folder structure

We should all keep the same folder structure, and everybody works on different files on his branch.

When we want to push something to master, follow [this guide](https://jasonrudolph.com/blog/2009/02/25/git-tip-how-to-merge-specific-files-from-another-branch/) to selectively merge files.


```
├── README.md          <- The top-level README for developers using this project.
│   
├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- Reports, roadmap, TODOs, etc
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-raph-initial-data-exploration`.
│
├── references         <- Papers and stuff
│
├── plots              <- Generated graphics and figures to be used in reporting
│
│
└── src                <- Source code for use in this project.
    ├── __init__.py    <- It's an empty file, makes src a Python module so that you can `from src import function`
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    ├── hpsearch       <- Scripts to automatically search hyperparameters
    │   └── random_search.py
    │   
    ├── models         <- Scripts to train generic models and then use trained models to make
    │   │                 predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```
