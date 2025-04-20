from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test, label_encoder,X,Y):
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    cm = confusion_matrix(y_test, y_pred)

    with open("results/metrics.txt", "w") as f:
        f.write(report)
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("results/confusion_matrix.png")
    plt.close()
    
    importances = model.feature_importances_
    sns.barplot(x=importances, y=X.columns)
    plt.title("Feature Importance")
    plt.ylabel("features")
    plt.savefig("results/Feature Importance.png",bbox_inches="tight")
    plt.close()

