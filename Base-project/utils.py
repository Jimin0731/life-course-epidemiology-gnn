"""Script containing useful functions"""

import random
from pathlib import Path

import matplotlib.pyplot as plt

# from tsai.all import *
import numpy as np

# import torch
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def create_filename(*args, **kwargs):
    """Create a filename from the given keyword arguments."""
    parts = list(args)
    for name, value in kwargs.items():
        if isinstance(value, bool):
            parts.append(name if value else "")
        elif isinstance(value, float):
            parts.append(f"{name}{value:.2f}")
        else:
            parts.append(f"{name}{value}")
    return "_".join(parts)


def eval_metrics(y_pred, y_proba, Y_test, case_name, output_dir, save_name, randnum):
    random.seed(randnum)
    np.random.seed(randnum)

    precision, recall, _ = precision_recall_curve(Y_test, y_proba)

    acc = accuracy_score(Y_test, y_pred)
    prec = precision_score(Y_test, y_pred)
    rec = recall_score(Y_test, y_pred)
    fone = f1_score(Y_test, y_pred)
    auc_score = roc_auc_score(Y_test, y_proba)
    avprec = average_precision_score(Y_test, y_proba)
    brier = brier_score_loss(Y_test, y_proba)
    prauc = auc(recall, precision)

    print(f"{case_name} accuracy: {acc:.4f}")
    print(f"{case_name} precision: {prec:.4f}")
    print(f"{case_name} recall: {rec:.4f}")
    print(f"{case_name} f1: {fone:.4f}")
    print(f"{case_name} auc: {auc_score:.4f}")
    print(f"{case_name} avprec: {avprec:.4f}")
    print(f"{case_name} brier: {brier:.4f}")
    print(f"{case_name} pr_auc: {prauc:.4f}")

    # Confusion Matrix
    cm2 = confusion_matrix(Y_test, y_pred)
    true_neg, false_pos, false_neg, true_pos = cm2.ravel()
    print(case_name)
    print("{:<40} {:.6f}".format("Y 0, predicted 0 (true negatives)", true_neg))
    print("{:<40} {:.6f}".format("Y 0, predicted 1 (false positives)", false_pos))
    print("{:<40} {:.6f}".format("Y 1, predicted 0 (false negatives)", false_neg))
    print("{:<40} {:.6f}".format("Y 1, predicted 1 (true positives)", true_pos))

    # Plotting ROC and PR curves
    RocCurveDisplay.from_predictions(Y_test, y_proba)
    Path(output_dir / "model_results/calibration/").mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_dir
        / "".join(
            [
                "model_results/calibration/output_",
                save_name,
                case_name,
                "_roc_curve.png",
            ]
        )
    )
    plt.clf()

    PrecisionRecallDisplay.from_predictions(Y_test, y_proba)
    plt.savefig(
        output_dir
        / "".join(
            [
                "model_results/calibration/output_",
                save_name,
                case_name,
                "_precrec_curve.png",
            ]
        )
    )
    plt.clf()

    prob_true, prob_pred = calibration_curve(Y_test, y_pred, n_bins=10)

    # Plot the Probabilities Calibrated curve
    plt.plot(prob_pred, prob_true, marker="o", linewidth=1, label="Model")

    # Plot the Perfectly Calibrated by Adding the 45-degree line to the plot
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly Calibrated")

    # Set the title and axis labels for the plot
    plt.title("Probability Calibration Curve")
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Probability")

    # Add a legend to the plot
    plt.legend(loc="best")

    # Show the plot
    plt.savefig(
        output_dir
        / "".join(
            [
                "model_results/calibration/output_",
                save_name,
                case_name,
                "_calibration.png",
            ]
        )
    )
    plt.clf()
    df_pp = pd.DataFrame(prob_pred)
    df_pt = pd.DataFrame(prob_true)
    df_pp.to_csv(
        output_dir
        / "".join(
            [
                "model_results/calibration/output_",
                save_name,
                case_name,
                "_calibration_prob_pred.csv",
            ]
        ),
        index=False,
    )
    df_pt.to_csv(
        output_dir
        / "".join(
            [
                "model_results/calibration/output_",
                save_name,
                case_name,
                "_calibration_prob_true.csv",
            ]
        ),
        index=False,
    )

    eval_metrics_out = [
        acc,
        prec,
        rec,
        fone,
        auc_score,
        avprec,
        brier,
        prauc,
        true_neg,
        false_pos,
        false_neg,
        true_pos,
    ]

    return eval_metrics_out


def threshold_func(Y_test, y_proba, output_dir, save_name, randnum):
    # Calculate metrics on calibrated probabilities

    random.seed(randnum)
    np.random.seed(randnum)

    precision, recall, thresholds = precision_recall_curve(Y_test, y_proba)

    # Same thresholding and metrics calculation process
    precision = precision[:-1]
    recall = recall[:-1]
    denom = precision + recall
    fscore = np.divide(
        2 * precision * recall,
        denom,
        out=np.zeros_like(denom),
        where=denom != 0,
    )
    ix = np.argmax(fscore)
    best_threshold = thresholds[ix]
    best_f1_score = fscore[ix]
    y_pred_thres = (y_proba >= best_threshold).astype(int)

    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label="Precision-Recall curve")
    plt.scatter(
        recall[ix],
        precision[ix],
        color="red",
        label=f"Best Threshold: {best_threshold:.2f}",
        marker="o",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()
    plt.savefig(
        output_dir
        / "".join(
            [
                "model_results/calibration/output_",
                save_name,
                "_threshold.png",
            ]
        )
    )
    plt.clf()
    return y_pred_thres, best_threshold, best_f1_score
