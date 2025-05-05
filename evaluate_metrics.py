from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_all_models(y_test, X_test, results):
    print("----- Classification Metrics -----\n")

    for model_name in results["original"].keys():
        print(f"🔹 Model: {model_name}")

        y_pred_original = results["original"][model_name]["y_pred"]
        y_pred_augmented = results["augmented"][model_name]["y_pred"]

        model_original = results["original"][model_name]["model"]
        model_augmented = results["augmented"][model_name]["model"]

        try:
            roc_auc_orig = roc_auc_score(y_test, model_original.predict_proba(X_test)[:, 1])
            roc_auc_aug = roc_auc_score(y_test, model_augmented.predict_proba(X_test)[:, 1])
        except:
            roc_auc_orig = roc_auc_aug = None

        metrics = {
            "Accuracy": (accuracy_score(y_test, y_pred_original), accuracy_score(y_test, y_pred_augmented)),
            "Precision": (precision_score(y_test, y_pred_original, average="weighted"),
                          precision_score(y_test, y_pred_augmented, average="weighted")),
            "Recall": (recall_score(y_test, y_pred_original, average="weighted"),
                       recall_score(y_test, y_pred_augmented, average="weighted")),
            "F1 Score": (f1_score(y_test, y_pred_original, average="weighted"),
                         f1_score(y_test, y_pred_augmented, average="weighted")),
        }

        if roc_auc_orig is not None:
            metrics["ROC AUC"] = (roc_auc_orig, roc_auc_aug)

        for metric, (before, after) in metrics.items():
            print(f"{metric:<12} BEFORE: {before:.4f} | AFTER: {after:.4f}")
        print()
