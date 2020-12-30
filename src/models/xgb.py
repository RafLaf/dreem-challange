from xgboost import XGBClassifier

def gen_model(**kwargs):
    return XGBClassifier(**kwargs)
