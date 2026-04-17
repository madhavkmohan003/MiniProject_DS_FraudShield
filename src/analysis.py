from sklearn.metrics import classification_report, confusion_matrix

def evaluate(y_test, y_pred):
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))