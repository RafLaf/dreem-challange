from sklearn.svm import SVC

def gen_model(C=20):
    return SVC(C=C)
