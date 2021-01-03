import yaml
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from utils.load_features import *

random_grid = {
        'learning_rate': [0.01 * i for i in range(4, 8, 2)],
        'n_estimators': [200 * i for i in range(4, 8, 2)],
        'min_child_weight': [5],
        'gamma': [1.5, 2],
        'subsample': [0.9],
        'colsample_bytree': [0.9],
        'max_depth': [3]
        }




X = read('train', RC=True, PATH="../../data/")
X = scale(X)
y=readlabel(PATH="../../data/")

xgb = XGBClassifier(objective='binary:logistic', use_label_encoder=False, nthread=1, eval_metric='mlogloss')
xgb_random = RandomizedSearchCV(estimator=xgb, scoring='f1_weighted', param_distributions=random_grid, n_iter=8, cv=5, verbose=2, random_state=42, n_jobs=-1)

xgb_random.fit(X, y)

print(xgb_random.best_params_)
with open('../params/xgb_raph.yml', 'w') as outfile:
    yaml.dump(xgb_random.best_params, outfile, default_flow_style=False)

results = pd.DataFrame(xgb_random.cv_results_)
results.to_csv("../../data/models/xgb_results_raph.csv")
