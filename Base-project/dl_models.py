"""Hyperparameter optimisation functions, model fitting functions and output analysis functions"""

import copy
import random
import timeit

import numpy as np
import pandas as pd
import torch
import torch.nn.modules.activation as tact
import tsai.calibration  # noqa # this import has a side effect of adding a method to the Learner class
from fastai.vision.all import (
    Categorize,
    EarlyStoppingCallback,
    FocalLossFlat,
)
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold
from tsai.data.all import (
    TSDataLoaders,
    TSDatasets,
    combine_split_data,
)
from tsai.data.validation import get_splits
from tsai.learner import ts_learner
from tsai.models.all import (
    InceptionTimePlus,
    LSTMAttention,
    MLSTM_FCNPlus,
    ResNetPlus,
)
from tsai.optuna import run_optuna_study

try:
    from lce import utils
except ModuleNotFoundError:
    import utils


def set_seed(seed):
    """Random seed for torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_seed):
    """Random seed for worker"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def run_dl_models(
    model_name,
    output_file,
    output_dir,
    data_name,
    stoc,
    X_trainvalid,
    Y_trainvalid,
    X_test,
    Y_test,
    run_hyper_opt,
    run_feature_importance,
    epochs,
    num_optuna_trials,
    randnum_split,
    randnum_train,
    randnum_stoc,
    folds,
    device,
    save_name,
    metrics,
    timestr,
):
    # Give the architecture for each model
    rem_list = [
        "alpha",
        "gamma",
        "lr_max",
        "ESPatience",
        "batch_size",
        "weight_decay",
    ]

    start1 = timeit.default_timer()

    if run_hyper_opt:
        # find the hyperparameters using optuna and cross-validation on train/valid
        trial = hyperopt(
            X_trainvalid,
            Y_trainvalid,
            epochs=epochs,
            num_optuna_trials=num_optuna_trials,
            model_name=model_name,
            randnum=randnum_split,
            folds=folds,
            device=device,
            save_name=save_name,
            metrics=metrics,
            output_dir=output_dir,
        )

        # formatting the selected hyperparameters to put in the model
        params = trial.params
        lr_max = params.get("lr_max")
        ESPatience = params.get("ESPatience")
        batch_size = params.get("batch_size")
        weight_decay = params.get("weight_decay")
        all_params = copy.copy(params)
        alpha = params.get("alpha")
        gamma = params.get("gamma")
        for key in rem_list:
            del params[key]

    else:
        # Pick model parameters
        params = {}

        # formatting the selected hyperparameters to put in the model
        lr_max = 1e-3
        ESPatience = 5
        batch_size = 32
        weight_decay = 0
        all_params = copy.copy(params)
        alpha = 0.5
        gamma = 2

    stop1 = timeit.default_timer()
    hype_time = stop1 - start1

    colnames = [
        "data_name",
        "model_name",
        "stoc",
        "randnum_train",
        "randnum_split",
        "randnum_stoc",
        "epochs",
        "trials",
        "folds",
        "ESPatience",
        "batch_size",
        "lr_max",
        "weight_decay",
        "train_time",
        "hype_time",
        "inf_time",
        "date",
        "best_threshold",
        "best_f1_score",
        "accuracy_nothres",
        "precision_nothres",
        "recall_nothres",
        "f1_nothres",
        "auc_nocal",
        "av_prec_nocal",
        "brier_nocal",
        "pr_auc_nocal",
        "true_neg_nothres",
        "false_pos_nothres",
        "false_neg_nothres",
        "true_pos_nothres",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "av_prec",
        "brier",
        "pr_auc",
        "true_neg",
        "false_pos",
        "false_neg",
        "true_pos",
    ]

    colnames.extend(list(all_params.keys()))
    output = pd.DataFrame(columns=colnames)

    # Split out the test set
    splits_9010 = get_splits(
        Y_trainvalid,
        valid_size=0.1,
        stratify=True,
        shuffle=True,
        test_size=0,
        show_plot=False,
        random_state=randnum_split,
    )

    # Retrain the model on train/test with the selected hyperparameters
    (
        train_time,
        inf_time,
        eval_metrics_out_nothres_nocal,
        eval_metrics_out_thres_cal,
        best_threshold,
        best_f1_score,
    ) = model_block(
        model_name=model_name,
        X=X_trainvalid,
        Y=Y_trainvalid,
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
        ESPatience=ESPatience,
        device=device,
        metrics=metrics,
        save_name=save_name,
        output_dir=output_dir,
        weight_decay=weight_decay,
        run_feature_importance=run_feature_importance,
    )

    # Formatting and saving the output
    outputs = (
        [
            data_name,
            model_name,
            stoc,
            randnum_train,
            randnum_split,
            randnum_stoc,
            epochs,
            num_optuna_trials,
            folds,
            ESPatience,
            batch_size,
            lr_max,
            weight_decay,
            train_time,
            hype_time,
            inf_time,
            timestr,
            best_threshold,
            best_f1_score,
        ]
        + eval_metrics_out_nothres_nocal
        + eval_metrics_out_thres_cal
    )

    outputs.extend(list(all_params.values()))
    output = pd.DataFrame([outputs], columns=colnames)
    output.to_csv(output_file, index=False)
    print(output)
    return output


def hyperopt(
    Xtrainvalid,
    Ytrainvalid,
    epochs,
    randnum,
    num_optuna_trials,
    model_name,
    device,
    folds,
    save_name,
    output_dir,
    metrics,
):
    """Hyperparameter optimisation and k-fold cross-validation for various models."""

    set_seed(randnum)

    # Model-specific hyperparameter ranges
    search_space = {
        "MLSTMFCN": {
            "kss": [
                [7, 5, 3],
                [3, 5, 7],
                [3, 5, 3],
                [5, 7, 5],
            ],
            "conv_layers": [
                [128, 256, 512],
                [64, 128, 256],
            ],
            "hidden_size": [60, 100, 120],
            "rnn_layers": [1, 2],
            "fc_dropout": [0.1, 0.3],
            "cell_dropout": [0.1, 0.3],
            "rnn_dropout": [0.1, 0.3],
        },
        "ResNet": {
            "nf": [32, 64, 128],
            "fc_dropout": [0.1, 0.3],
            "ks": [
                [7, 5, 3],
                [5, 7, 5],
                [3, 5, 7],
                [3, 5, 3],
            ],
        },
        "InceptionTime": {
            "nf": [32, 64, 128],
            "fc_dropout": [0.1, 0.3],
            "conv_dropout": [0.1, 0.3],
            "ks": [10, 20, 40],
        },
        "LSTMAttention": {
            "n_heads": [8, 16],
            "d_ff": [256, 512, 1024],
            "encoder_layers": [2, 3],
            "hidden_size": [128, 256],
            "fc_dropout": [0.1, 0.3],
            "rnn_dropout": [0.1, 0.3],
            "encoder_dropout": [0.1, 0.3],
            "rnn_layers": [1, 2],
        },
    }

    # General hyperparameters for all models
    general_search_space = {
        "lr_max": (1e-5, 1e-1),
        "batch_size": [32, 64, 128, 256],
        "ESPatience": [5, 10],
        "weight_decay": (1e-6, 1e-2),
        "alpha": (0.1, 0.5),
        "gamma": (1.01, 3),
    }

    def objective_cv(trial):
        """Objective function for cross-validation."""

        general_params = {
            key: (
                trial.suggest_float(key, *values)
                if isinstance(values, tuple)
                else trial.suggest_categorical(key, values)
            )
            for key, values in general_search_space.items()
        }

        def objective(trial):
            """Objective function for a single trial."""

            model_params = {
                key: trial.suggest_categorical(key, values)
                for key, values in search_space[model_name].items()
            }

            weights = torch.tensor(
                [general_params["alpha"], 1 - general_params["alpha"]],
                dtype=torch.float,
            ).to(device)

            if model_name == "MLSTMFCN":
                model = MLSTM_FCNPlus(dls.vars, dls.c, dls.len, **model_params)
            elif model_name == "ResNet":
                model = ResNetPlus(
                    dls.vars, dls.c, **model_params, act=tact.GELU
                )  # relu
            elif model_name == "InceptionTime":
                model = InceptionTimePlus(
                    dls.vars, dls.c, **model_params, act=tact.GELU
                )  # relu
            elif model_name == "LSTMAttention":
                model = LSTMAttention(dls.vars, dls.c, dls.len, **model_params)
            else:
                raise ValueError(f"Unknown model_name: {model_name}")

            model.to(device)

            learner = ts_learner(
                dls,
                model,
                metrics=metrics,
                loss_func=FocalLossFlat(
                    gamma=torch.tensor(general_params["gamma"]).to(device),
                    weight=weights,
                ),
                seed=randnum,
                cbs=[
                    EarlyStoppingCallback(patience=general_params["ESPatience"]),
                ],
            )

            learner.save("stage0")
            learner.fit_one_cycle(
                epochs,
                lr_max=general_params["lr_max"],
                wd=general_params["weight_decay"],
            )
            learner.save("stage1")

            return learner.recorder.values[-1][6]  # AUPRC

        scores = []

        # divide train data into folds
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=randnum)
        # skf.get_n_splits(Xtrainvalid, Ytrainvalid)

        # loop through each fold and fit hyperparameters
        for train_idx, valid_idx in skf.split(Xtrainvalid, Ytrainvalid):
            print("TRAIN:", train_idx, "VALID:", valid_idx)

            # select train and validation data
            Xtrain, Xvalid = Xtrainvalid[train_idx], Xtrainvalid[valid_idx]
            Ytrain, Yvalid = Ytrainvalid[train_idx], Ytrainvalid[valid_idx]

            X_combined, y_combined, stratified_splits = combine_split_data(
                [Xtrain, Xvalid], [Ytrain, Yvalid]
            )

            # prepare the data to go in the model
            tfms = [None, Categorize()]
            dsets = TSDatasets(
                X_combined,
                y_combined,
                tfms=tfms,
                splits=stratified_splits,
                inplace=True,
            )

            # prepare this data for the model (define batches etc)
            dls = TSDataLoaders.from_dsets(
                dsets.train,
                dsets.valid,
                bs=general_params["batch_size"],
                num_workers=0,
                device=device,
                worker_init_fn=seed_worker,
            )

            trial_score = objective(trial)
            scores.append(trial_score)

        return np.mean(scores)

    study = run_optuna_study(
        objective=objective_cv,
        seed=randnum,
        sampler=TPESampler(seed=randnum),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=1),
        study_name=save_name,
        direction="maximize",
        n_trials=num_optuna_trials,
        gc_after_trial=True,
        save_study=False,
    )

    print(study.trials_dataframe())
    filepathout = output_dir / "".join(
        [
            "model_results/optuna/optuna",
            save_name,
            ".csv",
        ]
    )
    entry = pd.DataFrame(study.trials_dataframe())
    entry.to_csv(filepathout, index=False)
    print(study.best_params)
    print(study.best_value)

    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    return study.best_trial


def model_block(
    model_name,
    X,
    Y,
    X_test,
    Y_test,
    splits,
    params,
    epochs,
    randnum,
    lr_max,
    alpha,
    gamma,
    batch_size,
    ESPatience,
    device,
    save_name,
    weight_decay,
    metrics,
    output_dir,
    run_feature_importance,
):
    """Fit the model on the train/test data with pre-trained hyperparameters"""

    set_seed(randnum)

    FLweights = [alpha, 1 - alpha]
    weights = torch.tensor(FLweights, dtype=torch.float).to(device)

    # prep the data for the model
    tfms = [None, Categorize()]
    dsets = TSDatasets(X, Y, tfms=tfms, splits=splits, inplace=True)

    # define batches
    dls = TSDataLoaders.from_dsets(
        dsets.train,
        dsets.valid,
        bs=batch_size,
        num_workers=0,
        device=device,
        shuffle=False,
        worker_init_fn=seed_worker,
    )

    # fit the model to the train/test data
    start = timeit.default_timer()

    if model_name == "MLSTMFCN":
        model = MLSTM_FCNPlus(dls.vars, dls.c, dls.len, **params)
    elif model_name == "ResNet":
        model = ResNetPlus(dls.vars, dls.c, **params, act=tact.GELU)
    elif model_name == "InceptionTime":
        model = InceptionTimePlus(dls.vars, dls.c, **params, act=tact.GELU)
    elif model_name == "LSTMAttention":
        model = LSTMAttention(dls.vars, dls.c, dls.len, **params)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    model.to(device)
    learn = ts_learner(
        dls,
        arch=model,
        metrics=metrics,
        loss_func=FocalLossFlat(gamma=torch.tensor(gamma).to(device), weight=weights),
        seed=randnum,
        cbs=[EarlyStoppingCallback(patience=ESPatience)],
        wd=weight_decay,
    )
    learn.save(
        "".join(
            [
                save_name,
                "_finalmodel_stage0",
            ]
        )
    )

    learn.fit_one_cycle(epochs, lr_max)
    learn.save(
        "".join(
            [
                save_name,
                "_finalmodel_stage1",
            ]
        )
    )

    stop = timeit.default_timer()
    train_time = stop - start

    start2 = timeit.default_timer()

    valid_dl = learn.dls.valid

    # obtain probability scores, predicted values and targets
    test_ds = valid_dl.dataset.add_test(X_test, Y_test)
    test_dl = valid_dl.new(test_ds)

    test_probas_c, test_targets_c, test_preds_c = learn.get_preds(
        dl=test_dl, with_decoded=True, save_preds=None, save_targs=None
    )

    # Caluclate evaluation metrics
    eval_metrics_out_nothres_nocal = utils.eval_metrics(
        test_preds_c,
        test_probas_c[:, 1],
        Y_test,
        "no_threshold_or_calibration",
        output_dir,
        save_name,
        randnum=randnum,
    )

    learn.calibrate_model(lr=0.0001, strategy="quantile")
    calibrated_model = learn.calibrated_model

    # Convert data to a torch tensor if it's not already one
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Access the underlying model from the ModelWithTemperature wrapper
    underlying_model = calibrated_model.model

    # Move the tensor to the same device as the underlying model
    X_test_tensor = X_test_tensor.to(next(underlying_model.parameters()).device)

    # Set the underlying model to evaluation mode
    underlying_model.eval()

    # Get the raw logits from the model
    with torch.no_grad():
        logits = calibrated_model(X_test_tensor)  # Use the calibrated model directly

        # Apply softmax to get the probability estimates
        probabilities = torch.nn.functional.softmax(logits, dim=1)

    # Convert the probabilities tensor to a numpy array
    probabilities_np = probabilities.cpu().numpy()

    y_proba_calibrated = probabilities_np[:, 1]

    # Alter threshold to maximise f1 score
    pred_thres, best_threshold, best_f1_score = utils.threshold_func(
        Y_test=Y_test,
        y_proba=y_proba_calibrated,
        output_dir=output_dir,
        save_name=save_name,
        randnum=randnum,
    )

    # Caluclate evaluation metrics
    eval_metrics_out_thres_cal = utils.eval_metrics(
        pred_thres,
        y_proba_calibrated,
        Y_test,
        "with_threshold_and_calibration",
        output_dir,
        save_name,
        randnum=randnum,
    )

    stop2 = timeit.default_timer()
    inf_time = stop2 - start2

    if run_feature_importance:
        try:
            from lce import explr
        except ModuleNotFoundError:
            import explr

        explr.explain_func(
            learn=learn,
            X_test=X_test,
            Y_test=Y_test,
            output_dir=output_dir,
            randnum=randnum,
            dls=dls,
            batch_size=batch_size,
            save_name=save_name,
        )

    return (
        train_time,
        inf_time,
        eval_metrics_out_nothres_nocal,
        eval_metrics_out_thres_cal,
        best_threshold,
        best_f1_score,
    )
