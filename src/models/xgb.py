from sklearn.ensemble import GradientBoostingClassifier

def gen_model(**kwargs):
    return GradientBoostingClassifier(**kwargs)
