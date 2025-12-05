import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

def plot_confusion_matrix(cm, classes, title="Confusion matrix", normalize=False):
    """
    Plots a confusion matrix using matplotlib.
    If normalize=True, also shows normalized values (row-wise).
    """
    if normalize:
        # avoid division by zero
        row_sums = cm.sum(axis=1, keepdims=True)
        norm_cm = np.divide(cm, row_sums, where=(row_sums != 0))
        display = norm_cm
    else:
        display = cm

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(display, interpolation='nearest', aspect='auto')
    ax.set_title(title)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    # Annotate cells with counts (and normalized values if requested)
    thresh = display.max() / 2.0
    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            if normalize:
                txt = f"{cm[i, j]:d}\n({display[i, j]:.2f})"
            else:
                txt = f"{cm[i, j]:d}"
            ax.text(j, i, txt,
                    ha="center", va="center",
                    fontsize=10,
                    color="white" if display[i, j] > thresh else "black")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


def train_model(x_train, y_train, x_test, y_test,
                param_grid=None, n_iter=20, cv=5, random_state=42, n_jobs=-1):
    # default param grid (if not provided)
    if param_grid is None:
        param_grid = {
            "num_leaves": [15, 31, 63],
            "max_depth": [-1, 5, 10],
            "min_child_samples": [10, 20, 40],
            "min_child_weight": [1e-4, 1e-3, 1e-2],
            "class_weight": [None, "balanced"],
            "learning_rate": [0.01, 0.1, 0.2],
            "n_estimators": [100, 200, 500],
            "reg_alpha": [0, 0.1, 0.4, 0.8],
            "reg_lambda": [0, 0.1, 0.4, 0.8]
        }

    base_model = LGBMClassifier(random_state=random_state)

    ran_m = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring='recall',
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1
    )

    # Fit tuner
    ran_m.fit(x_train, y_train)

    print("Best params found:")
    print(ran_m.best_params_)
    print(f"Best CV {ran_m.scoring if hasattr(ran_m, 'scoring') else 'score'}: {ran_m.best_score_:.4f}")
    print("-" * 40)

    # Use best estimator for predictions
    best = ran_m.best_estimator_

    # Predictions
    y_train_pred = best.predict(x_train)
    y_test_pred = best.predict(x_test)

    # Probabilities (for roc auc) if available
    y_train_proba = None
    y_test_proba = None
    try:
        y_test_proba = best.predict_proba(x_test)[:, 1]  # may fail for multi-class
        y_train_proba = best.predict_proba(x_train)[:, 1]
    except Exception:
        # no probability support or multiclass
        pass

    # Utility to print metrics succinctly
    def print_metrics(name, y_true, y_pred, y_proba=None):
        print(f"--- {name} metrics ---")
        print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
        # handle binary vs multiclass for average
        average = "binary" if len(np.unique(y_true)) == 2 else "macro"
        print(f"Precision: {precision_score(y_true, y_pred, average=average, zero_division=0):.4f}")
        print(f"Recall   : {recall_score(y_true, y_pred, average=average, zero_division=0):.4f}")
        print(f"F1-score : {f1_score(y_true, y_pred, average=average, zero_division=0):.4f}")
        # ROC AUC if probabilities available and binary
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                print(f"ROC AUC  : {roc_auc_score(y_true, y_proba):.4f}")
            except Exception:
                pass
        print("Classification report:")
        print(classification_report(y_true, y_pred, zero_division=0))
        print()

    # Print metrics
    print_metrics("Train", y_train, y_train_pred, y_train_proba)
    print_metrics("Test", y_test, y_test_pred, y_test_proba)

    # Confusion matrices
    labels = np.unique(np.concatenate([y_train, y_test]))
    cm_test = confusion_matrix(y_test, y_test_pred, labels=labels)
    plot_confusion_matrix(cm_test, classes=labels, title="Test set confusion matrix (counts)", normalize=False)
    plot_confusion_matrix(cm_test, classes=labels, title="Test set confusion matrix (normalized)", normalize=True)

    return ran_m  # return search object for further inspection
