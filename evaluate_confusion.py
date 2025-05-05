import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def show_all_confusion_matrices(y_test, results):
    for model_name in results["original"].keys():
        print(f"🔸 Confusion Matrix - {model_name}")

        y_pred_orig = results["original"][model_name]["y_pred"]
        y_pred_aug = results["augmented"][model_name]["y_pred"]

        model_orig = results["original"][model_name]["model"]
        model_aug = results["augmented"][model_name]["model"]

        # Original
        cm_orig = confusion_matrix(y_test, y_pred_orig)
        disp_orig = ConfusionMatrixDisplay(cm_orig, display_labels=model_orig.classes_)
        print("  🔻 Original Data:")
        disp_orig.plot(cmap="Reds")
        plt.show()

        # Augmented
        cm_aug = confusion_matrix(y_test, y_pred_aug)
        disp_aug = ConfusionMatrixDisplay(cm_aug, display_labels=model_aug.classes_)
        print("  🟢 Augmented Data:")
        disp_aug.plot(cmap="Blues")
        plt.show()

        fn_orig = cm_orig[1, 0] if cm_orig.shape == (2, 2) else "N/A"
        fn_aug = cm_aug[1, 0] if cm_aug.shape == (2, 2) else "N/A"
        print(f"False Negatives BEFORE: {fn_orig} | AFTER: {fn_aug}\n")
