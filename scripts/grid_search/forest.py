import yaml
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from load_features import *

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt', 'log2']
criterion = ['gini', 'entropy']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]


random_grid = {'n_estimators': n_estimators,
               'criterion': criterion,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


X = read('train')
X = scale(X)
y=readlabel()

rf = RandomForestClassifier() 
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

rf_random.fit(X, y)

print(rf_random.best_params_)
with open('../../data/models/best_rf_params.yml', 'w') as outfile:
    yaml.dump(rf_random.best_params, outfile, default_flow_style=False)

pd.DataFrame.drop()
results = pd.DataFrame(rf_random.cv_results_)
results.to_csv("../../data/models/rf_cv_results.csv")
