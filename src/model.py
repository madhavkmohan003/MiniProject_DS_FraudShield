from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

def train_model(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_res, y_res)

    return model