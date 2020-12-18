from sklearn.linear_model import LogisticRegression

def gen_model(solver='lbfgs'):
    return LogisticRegression(solver=solver)
