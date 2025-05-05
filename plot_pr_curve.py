import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_precision_recall_curves(y_test, X_test, results):
    for model_name in results["original"].keys():
        model_orig = results["original"][model_name]["model"]
        model_aug = results["augmented"][model_name]["model"]

        try:
            y_proba_orig = model_orig.predict_proba(X_test)[:, 1]
            y_proba_aug = model_aug.predict_proba(X_test)[:, 1]
        except AttributeError:
            print(f"⚠️ {model_name} does not support `predict_proba()`. Skipping.")
            continue

        prec_orig, rec_orig, _ = precision_recall_curve(y_test, y_proba_orig)
        pr_auc_orig = average_precision_score(y_test, y_proba_orig)

        prec_aug, rec_aug, _ = precision_recall_curve(y_test, y_proba_aug)
        pr_auc_aug = average_precision_score(y_test, y_proba_aug)

        plt.figure(figsize=(8, 5))
        plt.plot(rec_orig, prec_orig, linestyle='--', label=f"Original (PR-AUC = {pr_auc_orig:.4f})")
        plt.plot(rec_aug, rec_aug, label=f"Augmented (PR-AUC = {pr_auc_aug:.4f})", color="green")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - {model_name}")
        plt.legend()
        plt.grid(True)
        plt.show()
