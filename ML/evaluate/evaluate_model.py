import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

def evaluate_model(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, probs)
    plt.figure()
    plt.plot(recall, precision, label=f"AP={average_precision_score(y_test, probs):.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig("reports/pr_curve.png")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test, probs):.2f}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("reports/roc_curve.png")