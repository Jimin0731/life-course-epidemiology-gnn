"""Script containing hyperparameter optimisation functions, model fitting functions and output analysis functions"""

import sys
from ast import literal_eval

import dl_models
import load_data
import numpy as np
import pandas as pd
from fastai.metrics import BrierScore, F1Score, RocAucBinary, accuracy
from fastai.vision.all import *
from tsai.all import *
from tsai.data.validation import get_splits
from tsai.models.InceptionTimePlus import InceptionTimePlus

### Model arguments, some of which are set in command line

# Arguments related to system setup
FILEPATH = "/home/DIDE/smishra/Simulations/"  ### change
DEVICE = int(sys.argv[1])  #'cuda' if torch.cuda.is_available() else 'cpu'

# Arguments related to the data set used
DATA_NAME = sys.argv[2]
MODEL_NAME = sys.argv[3]
STOC = float(sys.argv[4])

# Arguments related to the random seeds
RANDNUM_TRAIN = int(sys.argv[5])
RANDNUM_SPLIT = 3
RANDNUM_STOC = 4

# Arguments relating to model setup and optimisation
IMP = "False"


def run(
    filepath,
    device,
    data_name,
    model_name,
    stoc,
    randnum_train,
    randnum_split,
    randnum_stoc,
    imp,
):
    """Function to get explainability insights from a fitted model

    Args:
        filepath (string): Gives the location of the data
        device (int): For cuda, is 0 or 1 to give the device to run
        data_name (string): Name of the simulated data.
        model_name (string): Name of the DL model, either ResNet, LSTMAttention, MLSTMFCN, InceptionTime or LR
        stoc (float): The proportion of noise in the data, between 0 and 1
        randnum_train (int): Seed used to control optimisation process and model weight initialisation
        randnum_split (int): Seed used to split the data into test and train groups
        randnum_stoc (int): Seed used to control which data is switched when noisy
        imp (bool): Whether feature importance will be conducted

    """

    # set random seeds for reproducibility
    load_data.set_random_seeds(randnum_train)
    print(data_name)

    # Load the input data (same for all outputs)
    X_raw = np.load("".join([filepath, "Simulated_data/", data_name, "_X.npy"])).astype(
        np.float32
    )

    # Load the output data (specific to LCP)
    Y_raw = np.squeeze(
        np.load("".join([filepath, "Simulated_data/", data_name, "_YH.npy"]))
    )
    y_raw = Y_raw[:, np.shape(Y_raw)[1] - 1]

    ## split out the test set
    splits = get_splits(
        y_raw,
        valid_size=0.2,
        stratify=True,
        shuffle=True,
        test_size=0,
        show_plot=False,
        random_state=randnum_split,
    )
    X_trainvalid, X_test = X_raw[splits[0]], X_raw[splits[1]]
    Y_trainvalid, Y_test = y_raw[splits[0]], y_raw[splits[1]]

    # Add noise to the data
    if stoc > 0:
        Y_trainvalid_stoc = load_data.add_noise(
            Y_trainvalid, stoc=stoc, randnum=randnum_stoc
        )
    else:
        Y_trainvalid_stoc = Y_trainvalid

    # Giving the filepath for the output
    savename = "".join(
        [
            "explr_",
            data_name,
            "_stoc",
            str(int(stoc * 100)),
            "_",
            model_name,
            "_randsp",
            str(int(randnum_split)),
            "_randtr",
            str(int(randnum_train)),
            "_fixagain2",
        ]
    )
    filepathout = "".join(
        [
            filepath,
            "Simulations/model_results/outputCVL_",  ## change
            savename,
            ".csv",
        ]
    )

    print(f"data= {data_name}; model name = {model_name}; stoc= {stoc}")

    # Evaluation metrics output when fitting the model
    metrics = [accuracy, F1Score(), RocAucBinary(), BrierScore(), APScoreBinary()]

    # Table of best fitting models
    params_mat = pd.read_csv("Plotting/plot_data.csv")

    # formatting the selected hyperparameters to put in the model
    lr_max = 1e-3
    es_patience = 2
    weight_decay = 0

    params_row = params_mat.loc[
        (params_mat["data"] == data_name)
        & (params_mat["model"] == model_name)
        & (params_mat["stoc"] == int(stoc * 100))
    ]

    batch_size = int(params_row["batch_size"].values[0])
    epochs = 10  # int(params_row["epochs"].values[0])
    alpha = params_row["alpha"].values[0]
    gamma = params_row["gamma"].values[0]
    print(f"alpha={alpha}, gamma={gamma}, batch_size={batch_size}, epochs={epochs}")

    if np.isnan(batch_size):
        batch_size = 32
    if np.isnan(alpha):
        alpha = 0.2
    if np.isnan(gamma):
        gamma = 2
    if np.isnan(epochs):
        epochs = 10

    if model_name == "ResNet":
        arch = ResNetPlus
        params = {
            "nf": int(params_row["nf"].values[0]),
            "ks": literal_eval(params_row["ks"].values[0]),
            "fc_dropout": params_row["fc_dropout"].values[0],
        }
        if np.isnan(params["nf"]):
            params["nf"] = 32
        # if np.isnan(params['ks']):
        #    params['ks'] = [7,5,3]
        if np.isnan(params["fc_dropout"]):
            params["fc_dropout"] = 0.2

    elif model_name == "InceptionTime":
        arch = InceptionTimePlus
        params = {
            "nf": int(params_row["nf"].values[0]),
            "ks": int(params_row["ks"].values[0]),
            "fc_dropout": params_row["fc_dropout"].values[0],
            "conv_dropout": params_row["conv_dropout"].values[0],
        }
        if np.isnan(params["nf"]):
            params["nf"] = 32
        if np.isnan(params["ks"]):
            params["ks"] = 40
        if np.isnan(params["fc_dropout"]):
            params["fc_dropout"] = 0.2
        if np.isnan(params["conv_dropout"]):
            params["conv_dropout"] = 0.2

    elif model_name == "MLSTMFCN":
        arch = MLSTM_FCNPlus
        params = {
            "kss": literal_eval(
                params_row["kss"].values[0]
            ),  # trial.suggest_categorical('kss', choices=kscombinations),
            "conv_layers": literal_eval(
                params_row["conv_layers"].values[0]
            ),  # trial.suggest_categorical('conv_layers', choices=combinations),
            "hidden_size": int(
                params_row["hidden_size"].values[0]
            ),  # trial.suggest_categorical('hidden_size', [60,80,100,120]),
            "rnn_layers": int(
                params_row["rnn_layers"].values[0]
            ),  # trial.suggest_categorical('rnn_layers', [1,2,3]),
            "fc_dropout": params_row["fc_dropout"].values[
                0
            ],  # 0.2,#trial.suggest_float('fc_dropout', 0.1, 0.5),
            "cell_dropout": params_row["cell_dropout"].values[
                0
            ],  # 0.2,#trial.suggest_float('cell_dropout', 0.1, 0.5),
            "rnn_dropout": params_row["rnn_dropout"].values[
                0
            ],  # 0.2,#trial.suggest_float('rnn_dropout', 0.1, 0.5),
        }
        # if np.isnan(params['kss']):
        #    params['kss'] = [7,5,3]
        # if np.isnan(params['conv_layers']):
        #    params['conv_layers'] = [256,128,128]
        if np.isnan(params["hidden_size"]):
            params["hidden_size"] = 100
        if np.isnan(params["rnn_layers"]):
            params["rnn_layers"] = 2
        if np.isnan(params["fc_dropout"]):
            params["fc_dropout"] = 0.2
        if np.isnan(params["cell_dropout"]):
            params["cell_dropout"] = 0.2
        if np.isnan(params["rnn_dropout"]):
            params["rnn_dropout"] = 0.2

    elif model_name == "LSTMAttention":
        arch = LSTMAttention
        params = {
            "n_heads": int(
                params_row["n_heads"].values[0]
            ),  # trial.suggest_categorical('n_heads', [8,12,16]),#'n_heads': trial.suggest_categorical('n_heads', [8,16,32]),
            "d_ff": int(
                params_row["d_ff"].values[0]
            ),  # trial.suggest_categorical('d_ff', [256,512,1024,2048,4096]),#2564096#'d_ff': trial.suggest_categorical('d_ff', [64,128,256]),
            "encoder_layers": int(
                params_row["encoder_layers"].values[0]
            ),  # trial.suggest_categorical('encoder_layers', [2,3,4]),
            "hidden_size": int(
                params_row["hidden_size"].values[0]
            ),  # trial.suggest_categorical('hidden_size', [32,64,128]),
            "rnn_layers": int(
                params_row["rnn_layers"].values[0]
            ),  # trial.suggest_categorical('rnn_layers', [1,2,3]),
            "fc_dropout": params_row["fc_dropout"].values[
                0
            ],  # trial.suggest_float('fc_dropout', 0.1, 0.5),
            "encoder_dropout": params_row["encoder_dropout"].values[
                0
            ],  # trial.suggest_float('encoder_dropout', 0.1, 0.5),
            "rnn_dropout": params_row["rnn_dropout"].values[
                0
            ],  # trial.suggest_float('rnn_dropout', 0.1, 0.5),
        }
        if np.isnan(params["n_heads"]):
            params["n_heads"] = 16
        if np.isnan(params["d_ff"]):
            params["d_ff"] = 256
        if np.isnan(params["hidden_size"]):
            params["hidden_size"] = 32
        if np.isnan(params["encoder_layers"]):
            params["encoder_layers"] = 2
        if np.isnan(params["fc_dropout"]):
            params["fc_dropout"] = 0.2
        if np.isnan(params["encoder_dropout"]):
            params["encoder_dropout"] = 0.2
        if np.isnan(params["rnn_dropout"]):
            params["rnn_dropout"] = 0.2

    print(f"Model params = {params}")

    # Split out the test set
    splits_9010 = get_splits(
        Y_trainvalid_stoc,
        valid_size=0.1,
        stratify=True,
        shuffle=True,
        test_size=0,
        show_plot=False,
        random_state=randnum_split,
    )

    Xtrainvalid90 = X_trainvalid[splits_9010[0]]
    Ytrainvalid90 = Y_trainvalid_stoc[splits_9010[0]]
    Xtrainvalid10 = X_trainvalid[splits_9010[1]]
    Ytrainvalid10 = Y_trainvalid_stoc[splits_9010[1]]

    # Retrain the model on train/test with the selected hyperparameters
    (
        train_time,
        learner,
        acc,
        prec,
        rec,
        fone,
        auc,
        prc,
        brier,
        true_neg,
        false_pos,
        false_neg,
        true_pos,
        inf_time,
    ) = dl_models.model_block(
        model_name=model_name,
        arch=arch,
        X=X_trainvalid,
        Y=Y_trainvalid_stoc,
        X_test=X_test,
        Y_test=Y_test,
        splits=splits_9010,
        randnum=randnum_train,
        epochs=epochs,
        params=params,
        lr_max=lr_max,
        alpha=alpha,
        gamma=gamma,
        batch_size=batch_size,
        ESPatience=es_patience,
        device=device,
        metrics=metrics,
        savename=savename,
        filepath=filepath,
        weight_decay=weight_decay,
        imp=imp,
    )

    colnames = [
        "data_name",
        "model_name",
        "stoc",
        "randnum_train",
        "randnum_split",
        "randnum_stoc",
        "epochs",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "prc",
        "brier",
        "true_neg",
        "false_pos",
        "false_ng",
        "true_pos",
        "train_time",
        "inf_time",
        "batch_size",
        "alpha",
        "gamma",
    ]

    colnames.extend(list(params.keys()))
    output = pd.DataFrame(columns=colnames)

    # acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11=dl_models.test_results(learner,X_test,Y_test,filepath,savename)

    # Formatting and saving the output
    outputs = [
        data_name,
        model_name,
        stoc,
        randnum_train,
        randnum_split,
        randnum_stoc,
        epochs,
        batch_size,
        alpha,
        gamma,
        acc,
        prec,
        rec,
        fone,
        auc,
        prc,
        brier,
        true_neg,
        false_pos,
        false_neg,
        true_pos,
        train_time,
        inf_time,
    ]

    outputs.extend(list(params.values()))
    entry = pd.DataFrame([outputs], columns=colnames)
    output = pd.concat([output, entry], ignore_index=True)
    output.to_csv(filepathout, index=False)
    print(output)
    print(filepathout)

    return


if __name__ == "__main__":
    run(
        filepath=FILEPATH,
        device=DEVICE,
        data_name=DATA_NAME,
        model_name=MODEL_NAME,
        stoc=STOC,
        randnum_train=RANDNUM_TRAIN,
        randnum_split=RANDNUM_SPLIT,
        randnum_stoc=RANDNUM_STOC,
        imp=IMP,
    )
