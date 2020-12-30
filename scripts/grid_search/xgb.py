import yaml
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from scripts.load_features import *

loss = ['deviance', 'exponential']
learning_rate = [x for x in np.linspace(start = 0.001, stop = 0.501, num = 10)]
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
subsample = [x for x in np.linspace(start = 0.8, stop = 1.0, num = 3)]
max_depth = [int(x) for x in np.linspace(5, 105, num = 5)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]


random_grid = {'n_estimators': n_estimators,
               'loss': loss,
               'learning_rate': learning_rate,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}


X = read('train', RC=True, PATH="../../data/")
X = scale(X)
y=readlabel(PATH="../../data/")

xgb = GradientBoostingClassifier() 
xgb_random = RandomizedSearchCV(estimator = xgb, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

xgb_random.fit(X, y)

print(xgb_random.best_params_)
with open('../params/xgb.yml', 'w') as outfile:
    yaml.dump(xgb_random.best_params, outfile, default_flow_style=False)

results = pd.DataFrame(xgb_random.cv_results_)
results.to_csv("../../data/models/xgb_results.csv")
