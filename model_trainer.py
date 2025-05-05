from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
import pandas as pd

class ModelTrainer:
    def __init__(self):
        self.models = {
            "RandomForest": RandomForestClassifier(),
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "SVM": SVC(),
            "KNN": KNeighborsClassifier(),
            "GradientBoosting": GradientBoostingClassifier()
        }
        self.results = {
            "original": {},
            "augmented": {}
        }

    def train(self, X_train, y_train, X_test, y_test, synthetic_data=None):
        for name, model in self.models.items():
            # Train on original data
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            print(f"🔹 {name} - Original Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
            print("Classification Report:")
            print(classification_report(y_test, y_pred))

            self.results["original"][name] = {
                "model": model,
                "accuracy": acc,
                "f1_score": f1,
                "y_pred": y_pred
            }

        if synthetic_data is not None:
            X_synth = synthetic_data.drop(columns=['Class'])
            y_synth = synthetic_data['Class']
            augmented_X = pd.concat([X_train, X_synth], axis=0)
            augmented_y = pd.concat([y_train, y_synth], axis=0)

            for name, model in self.models.items():
                model.fit(augmented_X, augmented_y)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')

                print(f"🟢 {name} - Augmented Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
                print("Classification Report After Augmentation:")
                print(classification_report(y_test, y_pred))

                self.results["augmented"][name] = {
                    "model": model,
                    "accuracy": acc,
                    "f1_score": f1,
                    "y_pred": y_pred
                }

        return self.results
