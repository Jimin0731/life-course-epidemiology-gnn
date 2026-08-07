"""Script containing hyperparameter optimisation and model fitting for xgboost"""

import copy
import random
import timeit
from pathlib import Path

# from tsai.all import *
import numpy as np

# import torch
import optuna
import pandas as pd
import xgboost as xgb
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

# from fastai.vision.all import *
try:
    from lce import logistic_regression_model, utils
except ModuleNotFoundError:
    import logistic_regression_model
    import utils

try:
    from sklearn.frozen import FrozenEstimator
except ModuleNotFoundError:
    FrozenEstimator = None


def _fit_calibrator_prefit(model, X_train, y_train):
    """Fit a post-hoc calibrator while supporting both old/new sklearn APIs."""
    if FrozenEstimator is not None:
        calibrator = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    else:
        calibrator = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrator.fit(X_train, y_train)
    return calibrator


def run_xgboost(
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
    rem_list = ["ESPatience"]

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
        ESPatience = params.get("ESPatience")
        all_params = copy.copy(params)
        for key in rem_list:
            del params[key]

    else:
        # Pick model parameters
        params = {}
        ESPatience = 5  # params.get("ESPatience")
        all_params = copy.copy(params)

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
    train_idx, valid_idx = train_test_split(
        np.arange(len(Y_trainvalid)),
        test_size=0.1,
        random_state=randnum_split,
        stratify=Y_trainvalid if len(np.unique(Y_trainvalid)) > 1 else None,
    )
    splits_9010 = (train_idx, valid_idx)

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
        params=params,
        randnum=randnum_train,
        ESPatience=ESPatience,
        device=device,
        save_name=save_name,
        output_dir=output_dir,
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
    """Hyperparameter optimisation and k-fold cross-validation for XGBoost"""

    # Set the seed for reproducibility
    randnum_train = randnum

    random.seed(randnum_train)
    np.random.seed(randnum_train)

    def objective_cv(trial):
        """objective function enveloping the model objective function with cross-validation"""

        ESPatience = trial.suggest_categorical("ESPatience", [5, 10])
        # ESPatience = trial.suggest_categorical("ESPatience", [2, 5, 10, 20])

        # ✅ FIX: 클로저 문제 해결 - 데이터를 직접 인자로 받도록 수정
        def objective(trial, XLRtrain, Ytrain, XLRvalid, Yvalid):
            # Define the hyperparameters to tune
            param_grid = {
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "eta": trial.suggest_float("eta", 0.01, 0.3),  ## learning_rate
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "gamma": trial.suggest_float("gamma", 0, 1),
            }

            early_stop = xgb.callback.EarlyStopping(
                rounds=ESPatience,
                metric_name="map",
                data_name="validation_0",
                save_best=True,
            )

            # ✅ FIX: ZeroDivisionError 방지 (양성 샘플이 0인 fold 대응)
            n_pos = max(len(Ytrain[Ytrain == 1]), 1)
            scale_pos_weight = len(Ytrain[Ytrain == 0]) / n_pos

            # Initialize the XGBClassifier with the scale_pos_weight parameter
            model = XGBClassifier(
                **param_grid,
                tree_method="hist",
                callbacks=[early_stop],
                objective="binary:logistic",
                eval_metric="map",
                booster="gbtree",
                seed=randnum_train,
                scale_pos_weight=scale_pos_weight,
            )

            # Train the model with early stopping
            model.fit(
                XLRtrain,
                Ytrain,
                eval_set=[(XLRvalid, Yvalid)],
                verbose=True,
            )

            y_proba_test = model.predict_proba(XLRvalid)[:, 1]

            test_average_precision_score = average_precision_score(Yvalid, y_proba_test)

            return test_average_precision_score

        scores = []

        # divide train data into folds
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=randnum_train)
        skf.get_n_splits(Xtrainvalid, Ytrainvalid)

        # loop through each fold and fit hyperparameters
        for fold_idx, (train_idx, valid_idx) in enumerate(
            skf.split(Xtrainvalid, Ytrainvalid)
        ):
            print(f"Fold {fold_idx + 1} TRAIN:", train_idx, "VALID:", valid_idx)

            # select train and validation data
            Xtrain, Xvalid = Xtrainvalid[train_idx], Xtrainvalid[valid_idx]
            Ytrain, Yvalid = Ytrainvalid[train_idx], Ytrainvalid[valid_idx]

            XLRtrain, XLRvalid = logistic_regression_model.LM_reshape_func(
                Xtrain, Xvalid
            )

            # ✅ FIX: 데이터 인자 명시적으로 전달
            trial_score = objective(trial, XLRtrain, Ytrain, XLRvalid, Yvalid)
            scores.append(trial_score)

            trial.report(np.mean(scores), fold_idx)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(scores)

    # Create and run the study
    study = optuna.create_study(
        direction="maximize",
        study_name=save_name,
        sampler=TPESampler(seed=randnum_train, multivariate=True),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=1),
    )
    study.optimize(
        objective_cv,
        n_trials=num_optuna_trials,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    print(study.trials_dataframe())
    Path(output_dir / "model_results/optuna/").mkdir(parents=True, exist_ok=True)
    filepathout = output_dir / "".join(
        [
            "model_results/optuna/optunaoutput_",
            save_name,
            ".csv",
        ]
    )
    entry = pd.DataFrame(study.trials_dataframe())
    entry.to_csv(filepathout, index=False)
    print(study.best_params)
    print(study.best_value)

    print("Best trial: ")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print(" Params: ")
    for key, value in trial.params.items():
        print("   {}:{}".format(key, value))

    return trial


def model_block(
    model_name,
    X,
    Y,
    X_test,
    Y_test,
    splits,
    params,
    randnum,
    ESPatience,
    device,
    save_name,
    output_dir,
    run_feature_importance,
):
    """Will fit xgboost model with selected parameters"""

    random.seed(randnum)
    np.random.seed(randnum)

    X_train, X_valid = X[splits[0]], X[splits[1]]
    Y_train, Y_valid = Y[splits[0]], Y[splits[1]]

    # fit the model to the train/test data
    start = timeit.default_timer()

    XLRtrain, XLRvalid = logistic_regression_model.LM_reshape_func(X_train, X_valid)
    XLRtrainvalid, XLRtest = logistic_regression_model.LM_reshape_func(X, X_test)

    early_stop = xgb.callback.EarlyStopping(
        rounds=ESPatience,
        metric_name="map",
        data_name="validation_0",
        save_best=True,
    )

    # ✅ FIX: ZeroDivisionError 방지 (양성 샘플이 0인 경우 대응)
    n_pos = max(len(Y_train[Y_train == 1]), 1)
    scale_pos_weight = len(Y_train[Y_train == 0]) / n_pos

    # Initialize the XGBClassifier with the scale_pos_weight parameter
    model = XGBClassifier(
        **params,
        tree_method="hist",
        callbacks=[early_stop],
        objective="binary:logistic",
        eval_metric="map",
        booster="gbtree",
        seed=randnum,
        scale_pos_weight=scale_pos_weight,
    )

    print(params)

    # Train the model with early stopping
    model.fit(
        XLRtrain,
        Y_train,
        eval_set=[(XLRvalid, Y_valid)],
        verbose=True,
    )
    stop = timeit.default_timer()
    train_time = stop - start

    start2 = timeit.default_timer()

    y_pred_test = model.predict(XLRtest)
    y_proba_test = model.predict_proba(XLRtest)[:, 1]

    eval_metrics_out_nothres_nocal = utils.eval_metrics(
        y_pred=y_pred_test,
        y_proba=y_proba_test,
        Y_test=Y_test,
        case_name="no_threshold_or_calibration",
        output_dir=output_dir,
        save_name=save_name,
        randnum=randnum,
    )

    # Calibrate the model
    calibrated_model = _fit_calibrator_prefit(model, XLRtrainvalid, Y)
    y_proba_calibrated = calibrated_model.predict_proba(XLRtest)[:, 1]

    y_pred_thres, best_threshold, best_f1_score = utils.threshold_func(
        Y_test=Y_test,
        y_proba=y_proba_calibrated,
        output_dir=output_dir,
        save_name=save_name,
        randnum=randnum,
    )

    eval_metrics_out_thres_cal = utils.eval_metrics(
        y_pred=y_pred_thres,
        y_proba=y_proba_calibrated,
        Y_test=Y_test,
        case_name="with_threshold_and_calibration",
        output_dir=output_dir,
        save_name=save_name,
        randnum=randnum,
    )

    stop2 = timeit.default_timer()
    inf_time = stop2 - start2

    if run_feature_importance:
        try:
            from lce import explr
        except ModuleNotFoundError:
            import explr

        explr.explain_func_xgb(
            X=XLRtrainvalid,
            learn=model,
            output_dir=output_dir,
            save_name=save_name,
            Y_test=Y_test,
            randnum=randnum,
            X_test=X_test,
            XLRtest=XLRtest,
            n_people=np.shape(X)[0],
            n_features=np.shape(X)[1],
            n_time=np.shape(X)[2],
            y_pred=y_pred_thres,
        )

    return (
        train_time,
        inf_time,
        eval_metrics_out_nothres_nocal,
        eval_metrics_out_thres_cal,
        best_threshold,
        best_f1_score,
    )
