"""Function containing explainability functions, including SHAP and PFI"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
import tsai.analysis  # noqa # this import has a side effect of adding a method to the Learner class
from tsai import *


def explain_func(
    learn, X_test, Y_test, output_dir, randnum, dls, batch_size, save_name
):
    """Explainability function that carries out SHAP and PFI"""

    # Compute SHAP
    shap_func(
        dls,
        batch_size,
        learn=learn,
        output_dir=output_dir,
        save_name=save_name,
        Y_test=Y_test,
        randnum=randnum,
        X_test=X_test,
    )

    print("Done SHAP")

    # Compute PFI
    for metric_idx in [2, 4]:
        perm_imp(
            metric_idx=metric_idx,
            learn=learn,
            output_dir=output_dir,
            randnum=randnum,
            X_test=X_test,
            Y_test=Y_test,
            save_name=save_name,
        )

    return


def shap_func(dls, batch_size, learn, output_dir, save_name, Y_test, randnum, X_test):
    """Function to conduct SHAP"""

    train_batches = []

    for batch in dls.train:
        train_batches.append(batch)

    # Concatenate all the train batches to obtain the entire train dataset
    train_data = torch.cat([x[0] for x in train_batches], dim=0)

    # Create the GradientExplainer
    explainer = shap.GradientExplainer(
        learn.model.cpu(),
        train_data.cpu(),  # Use the entire train set
    )

    # Cast X_test to float32
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).cpu()

    # Get the SHAP values
    shap_values, indices = explainer.shap_values(
        X_test_tensor, rseed=randnum, ranked_outputs=2
    )

    # Save the shap values
    shap0 = shap_values[:, :, :, 0]
    shap1 = shap_values[:, :, :, 1]
    shap0 = shap0.reshape(shap0.shape[0], shap0.shape[1] * shap0.shape[2])
    shap1 = shap1.reshape(shap1.shape[0], shap1.shape[1] * shap1.shape[2])
    shap0 = pd.DataFrame(shap0)
    shap1 = pd.DataFrame(shap1)
    xtest = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

    xtest = pd.DataFrame(xtest)
    ytest = pd.DataFrame(np.array(Y_test))
    indices_out = pd.DataFrame(indices)

    Path("output/model_results/explr/").mkdir(parents=True, exist_ok=True)
    shap0.to_csv(
        output_dir
        / "".join(
            [
                "model_results/explr/output_",
                save_name,
                "_shap0.csv",
            ]
        ),
        index=False,
    )
    shap1.to_csv(
        output_dir
        / "".join(
            [
                "model_results/explr/output_",
                save_name,
                "_shap1.csv",
            ]
        ),
        index=False,
    )

    xtest.to_csv(
        output_dir
        / "".join(
            [
                "model_results/explr/output_",
                save_name,
                "_xtest.csv",
            ]
        ),
        index=False,
    )
    ytest.to_csv(
        output_dir
        / "".join(
            [
                "model_results/explr/output_",
                save_name,
                "_ytest.csv",
            ]
        ),
        index=False,
    )
    Path("output/model_results/explr/shap/values/").mkdir(parents=True, exist_ok=True)
    indices_out.to_csv(
        output_dir
        / "".join(
            [
                "model_results/explr/shap/values/output_",
                save_name,
                "_indices.csv",
            ]
        ),
        index=False,
    )

    return


def perm_imp(metric_idx, learn, output_dir, save_name, randnum, X_test, Y_test):
    """Function to conduct PFI (permutation feature importance)"""

    if metric_idx == 2:
        save_name = "".join([save_name, "_AUROC"])

    elif metric_idx == 4:
        save_name = "".join([save_name, "_AUPRC"])

    Path("output/model_results/explr/PFI/").mkdir(parents=True, exist_ok=True)
    learn.feature_importance(
        X=X_test,
        y=Y_test,
        key_metric_idx=metric_idx,
        save_df_path="".join(
            [
                str(output_dir),
                "/model_results/explr/PFI/",
                save_name,
                "feature_imp_full",
            ]
        ),
        random_state=randnum,
    )

    learn.step_importance(
        X=X_test,
        y=Y_test,
        key_metric_idx=metric_idx,
        save_df_path="".join(
            [str(output_dir), "/model_results/explr/PFI/", save_name, "step_imp_full"]
        ),
        random_state=randnum,
    )

    return


def explain_func_xgb(
    X,
    learn,
    output_dir,
    save_name,
    Y_test,
    randnum,
    X_test,
    XLRtest,
    n_people,
    n_features,
    n_time,
    y_pred,
):
    """Explainability function that carries out SHAP and PFI for XGB"""

    # Compute feature importance (in the flattened 2D space)
    importance = learn.get_booster().get_score(importance_type="gain")

    # Convert importance to a numpy array for easy manipulation
    importance_array = np.array(
        [importance.get(f"f{i}", 0) for i in range(XLRtest.shape[1])]
    )

    # Reshape importance back to 3D
    importance_3d = importance_array.reshape((n_features, n_time))

    # Plot importance in 3D space (e.g., as heatmaps)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(importance_3d, cmap="coolwarm", ax=ax)
    ax.set_title("Feature Importance Heatmap")

    plt.savefig(
        output_dir
        / "".join(
            [
                "model_results/explr/output_",
                save_name,
                "_3D_Feature_Importance.png",
            ]
        )
    )
    plt.clf()

    imp_out = pd.DataFrame(importance_3d)

    imp_out.to_csv(
        output_dir
        / "".join(
            [
                "model_results/explr/output_",
                save_name,
                "_Feature_Importance.csv",
            ]
        ),
        index=False,
    )

    # Compute SHAP values
    explainer = shap.TreeExplainer(learn)
    shap_values = explainer.shap_values(XLRtest)

    # Step 3: Reshape SHAP values back to 3D
    n_samples, n_features_flat = shap_values.shape
    n_timesteps = int(n_features_flat / n_features)  # Assuming this divides perfectly

    # Reshape SHAP values back to 3D (n_people, n_features, n_time)
    shap_values_3d = shap_values.reshape((n_samples, n_features, n_timesteps))

    # Example: Average SHAP values across people and visualize for each time point
    shap_mean = shap_values_3d.mean(axis=0)  # Average across the first axis (n_people)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(shap_mean, cmap="coolwarm", ax=ax)
    ax.set_title("Average SHAP Values Heatmap Across People")

    plt.savefig(
        output_dir
        / "".join(
            [
                "model_results/explr/output_",
                save_name,
                "_Averaged_SHAP_plotgbt.png",
            ]
        )
    )
    plt.clf()

    xtest = pd.DataFrame(XLRtest)
    ytest = pd.DataFrame(Y_test)
    indices_out = pd.DataFrame(y_pred)
    shap_out = pd.DataFrame(shap_values)
    # int_out = pd.DataFrame(shap_int)

    shap_out.to_csv(
        output_dir
        / "".join(
            [
                "model_results/explr/output_",
                save_name,
                "_shapxgb.csv",
            ]
        ),
        index=False,
    )

    xtest.to_csv(
        output_dir
        / "".join(
            [
                "model_results/explr/output_",
                save_name,
                "_xtest.csv",
            ]
        ),
        index=False,
    )
    ytest.to_csv(
        output_dir
        / "".join(
            [
                "model_results/explr/output_",
                save_name,
                "_ytest.csv",
            ]
        ),
        index=False,
    )
    indices_out.to_csv(
        output_dir
        / "".join(
            [
                "model_results/explr/output_",
                save_name,
                "_indices.csv",
            ]
        ),
        index=False,
    )

    return


def explain_func_LR(
    X,
    learn,
    output_dir,
    save_name,
    Y_test,
    randnum,
    X_test,
    XLRtest,
    n_people,
    n_features,
    n_time,
    y_pred,
):
    """Explainability function that carries out SHAP and PFI for XGB"""

    importance_array = learn.coef_

    # Reshape importance back to 3D
    importance_3d = importance_array.reshape((n_features, n_time))

    # Plot importance in 3D space (e.g., as heatmaps)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(importance_3d, cmap="coolwarm", ax=ax)
    ax.set_title("Feature Importance Heatmap")
    plt.show()
    plt.savefig(
        output_dir
        / "".join(
            [
                "model_results/explr/output_",
                save_name,
                "_LR coefs.png",
            ]
        )
    )
    plt.clf()

    # save shap values and interaction values

    xtest = pd.DataFrame(XLRtest)
    ytest = pd.DataFrame(Y_test)
    indices_out = pd.DataFrame(y_pred)
    shap_out = pd.DataFrame(importance_array)

    shap_out.to_csv(
        output_dir
        / "".join(
            [
                "model_results/explr/output_",
                save_name,
                "_shapxgb.csv",
            ]
        ),
        index=False,
    )

    xtest.to_csv(
        output_dir
        / "".join(
            [
                "model_results/explr/output_",
                save_name,
                "_xtest.csv",
            ]
        ),
        index=False,
    )
    ytest.to_csv(
        output_dir
        / "".join(
            [
                "model_results/explr/output_",
                save_name,
                "_ytest.csv",
            ]
        ),
        index=False,
    )
    indices_out.to_csv(
        output_dir
        / "".join(
            [
                "model_results/explr/output_",
                save_name,
                "_indices.csv",
            ]
        ),
        index=False,
    )

    return
