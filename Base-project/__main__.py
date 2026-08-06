"""File to control the overall running of the model"""

from pathlib import Path

import click
import numpy as np
import torch
from tsai.data.validation import get_splits

from lce import load_data, run_all_models

# Arguments related to system setup
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@click.command()
@click.option(
    "--data-dir",
    default="data/",
    help="Filepath to the data",
    type=click.Path(exists=True),
)
@click.option(
    "--output-dir",
    default="output/",
    help="Filepath to the model output",
    type=click.Path(exists=False),
)
@click.option(
    "--data-name",
    required=True,
    type=click.Choice(
        [
            "Repeats1",
            "Repeats2",
            "Repeats3",
            "Order1",
            "Order2",
            "Order3",
            "Timing1",
            "Timing2",
            "Timing3",
            "Timing4",
            "Period1",
            "Period2",
            "Period3",
            "Period4",
        ]
    ),
    help="Name of the simulated dataset",
)
@click.option(
    "--model-name",
    required=True,
    type=click.Choice(
        ["ResNet", "XGBoost", "LSTMAttention", "MLSTMFCN", "InceptionTime", "LR"]
    ),
    help="Name of the model",
)
@click.option(
    "--stochasticity",
    default=0,
    type=float,
    help="Proportion of noise in the data. Should be between 0 and 1.",
)
@click.option(
    "--train-seed",
    default=42,
    type=int,
    help="Seed for optimisation and weight initialisation",
)
@click.option(
    "--data-split-seed", default=42, type=int, help="Seed for splitting the data"
)
@click.option(
    "--stochasticity-seed",
    default=42,
    type=int,
    help="Seed for controlling noise in the data",
)
@click.option("--epochs", default=50, type=int, help="Number of epochs")
@click.option(
    "--num-optuna-trials",
    default=2,
    type=int,
    help="Number of hyperparameter optimisation trials",
)
@click.option(
    "--run-hyper-opt",
    default=False,
    is_flag=True,
    help="Whether hyperparameter optimisation will be conducted",
)
@click.option(
    "--run-feature-importance",
    default=False,
    is_flag=True,
    help="Whether feature importance will be conducted",
)
@click.option(
    "--folds", default=2, type=int, help="Number of folds in k-fold cross-validation"
)
@click.option(
    "--device",
    default=DEFAULT_DEVICE,
    help="Device to run the model on: 'cpu', 'cuda' or int specifying the GPU",
)
def main(
    data_dir,
    output_dir,
    data_name,
    model_name,
    stochasticity,
    train_seed,
    data_split_seed,
    stochasticity_seed,
    epochs,
    num_optuna_trials,
    run_hyper_opt,
    run_feature_importance,
    folds,
    device,
):
    """
    Function to train the model and provide insights into model performance
    """
    device = torch.device(device)

    # parse paths
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # make output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # set random seeds for reproducibility
    load_data.set_random_seeds(train_seed)
    print(data_name)

    # Load the model input data (same for all outputs)
    X_raw = np.load(data_dir / "data_X.npy").astype(np.float32)
    print(X_raw.shape)

    # Load the model output data (specific to LCP)
    Y_raw = np.squeeze(np.load(data_dir / "".join(["data_", data_name, "_Y.npy"])))
    y_raw = Y_raw[:, np.shape(Y_raw)[1] - 1]

    ## split out the test set
    splits = get_splits(
        y_raw,
        valid_size=0.2,
        stratify=True,
        shuffle=True,
        test_size=0,
        show_plot=False,
        random_state=data_split_seed,
    )
    X_trainvalid, X_test = X_raw[splits[0]], X_raw[splits[1]]
    Y_trainvalid, Y_test = y_raw[splits[0]], y_raw[splits[1]]

    # Add noise to the data
    if stochasticity > 0:
        Y_trainvalid_stoc = load_data.add_noise(
            Y_trainvalid, stoc=stochasticity, randnum=stochasticity_seed
        )
    else:
        Y_trainvalid_stoc = Y_trainvalid

    # Script to run model and collect outputs
    output = run_all_models.all_run(
        data_dir=data_dir,
        output_dir=output_dir,
        device=device,
        data_name=data_name,
        model_name=model_name,
        X_trainvalid=X_trainvalid,
        Y_trainvalid=Y_trainvalid_stoc,
        X_test=X_test,
        Y_test=Y_test,
        stoc=stochasticity,
        randnum_train=train_seed,
        randnum_split=data_split_seed,
        randnum_stoc=stochasticity_seed,
        epochs=epochs,
        num_optuna_trials=num_optuna_trials,
        run_hyper_opt=run_hyper_opt,
        run_feature_importance=run_feature_importance,
        folds=folds,
    )

    return output


if __name__ == "__main__":
    main()
