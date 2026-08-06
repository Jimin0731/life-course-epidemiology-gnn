"""Script to fit the logistic regression model"""

import copy
import random
import timeit
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold

try:
    from lce import utils
except ModuleNotFoundError:
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


def run_logistic_regression(
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
    # start timer
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
        all_params = copy.copy(params)

    else:
        # Pick model parameters
        params = {"C": 0.01}
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

    output = pd.DataFrame(columns=colnames)

    # fit the logistic regression model
    (
        train_time,
        inf_time,
        eval_metrics_out_nothres_nocal,
        eval_metrics_out_thres_cal,
        best_threshold,
        best_f1_score,
    ) = LRmodel_block(
        Xtrainvalid=X_trainvalid,
        Ytrainvalid=Y_trainvalid,
        Xtest=X_test,
        Ytest=Y_test,
        randnum=randnum_train,
        output_dir=output_dir,
        save_name=save_name,
        run_feature_importance=run_feature_importance,
        params=all_params,
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
    """Hyperparameter optimisation and k-fold cross-validation for LR"""

    # Set the seed for reproducibility
    randnum_train = randnum

    random.seed(randnum_train)
    np.random.seed(randnum_train)

    def objective_cv(trial):
        """objective function enveloping the model objective function with cross-validation"""

        def objective(trial):
            # Define the hyperparameters to tune
            param_grid = {
                "C": trial.suggest_categorical(
                    "C",
                    [
                        10**-5,
                        10**-4,
                        10**-3,
                        10**-2,
                        10**-1,
                        1,
                        10,
                        100,
                        1000,
                        10**4,
                        10**5,
                    ],
                ),
            }

            # Initialize the XGBClassifier with the scale_pos_weight parameter
            model = LogisticRegression(
                **param_grid,
                random_state=randnum_train,
                class_weight="balanced",
                solver="saga",
                max_iter=5000,
            )
            model.fit(XLRtrain, Ytrain)

            y_proba_test = model.predict_proba(XLRvalid)[:, 1]

            test_average_precision_score = average_precision_score(Yvalid, y_proba_test)

            return test_average_precision_score

        scores = []

        # divide train data into folds
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=randnum_train)
        skf.get_n_splits(Xtrainvalid, Ytrainvalid)

        # loop through each fold and fit hyperparameters
        for train_idx, valid_idx in skf.split(Xtrainvalid, Ytrainvalid):
            print("TRAIN:", train_idx, "VALID:", valid_idx)

            # select train and validation data
            Xtrain, Xvalid = Xtrainvalid[train_idx], Xtrainvalid[valid_idx]
            Ytrain, Yvalid = Ytrainvalid[train_idx], Ytrainvalid[valid_idx]

            XLRtrain, XLRvalid = LM_reshape_func(Xtrain, Xvalid)

            trial_score = objective(trial)
            scores.append(trial_score)

        return np.mean(scores)

    search_space = {
        "C": [
            10**-5,
            10**-4,
            10**-3,
            10**-2,
            10**-1,
            1,
            10,
            100,
            1000,
            10**4,
            10**5,
        ],
    }

    # Create and run the study
    study = optuna.create_study(
        direction="maximize",
        study_name=save_name,
        sampler=optuna.samplers.GridSampler(search_space, seed=randnum_train),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=1),
    )
    study.optimize(
        objective_cv,
        n_trials=num_optuna_trials,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    print(study.trials_dataframe())
    Path("output/model_results/optuna/").mkdir(parents=True, exist_ok=True)
    filepathout = output_dir / "".join(
        [
            "model_results/optuna/optunaoutput",
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


def LM_reshape_func(Xtrainvalid, Xtest):
    """Function to reshape data for the LR model by flattening the time/feature dimensions into one.

    Args:
        Xtrainvalid (nparray): Input training data.
        Xtest (nparray): Input testing data.

    Returns:
        nparray: Reshaped input training data.
        nparray: Reshaped input testing data.
    """

    X_LRtrainvalid = np.reshape(
        Xtrainvalid,
        (np.shape(Xtrainvalid)[0], np.shape(Xtrainvalid)[1] * np.shape(Xtrainvalid)[2]),
    )
    X_LRtest = np.reshape(
        Xtest, (np.shape(Xtest)[0], np.shape(Xtest)[1] * np.shape(Xtest)[2])
    )

    return X_LRtrainvalid, X_LRtest


def LRmodel_block(
    Xtrainvalid,
    Ytrainvalid,
    Xtest,
    Ytest,
    randnum,
    output_dir,
    save_name,
    run_feature_importance,
    params,
):
    """Function to fit and analyse the logistic regression model

    Args:
        Xtrainvalid (nparray): Training input data.
        Ytrainvalid (nparray): Training output data.
        Xtest (nparray): Testing input data.
        Ytest (nparray): Testing output data.
        randnum (int): Random seed for model training.
        output_dir (Path): Filepath for output saving.
        save_name (Path): Filename to save output.
        run_feature_importance
        params

    Returns:
        nparray: Values of evaluation metrics.
    """

    random.seed(randnum)
    np.random.seed(randnum)

    start = timeit.default_timer()

    # Flatten the data for the model
    X_LRtrain, X_LRtest = LM_reshape_func(Xtrainvalid, Xtest)

    # Fit the logistic regression model to the train data

    LRmodel = LogisticRegression(
        **params,
        penalty="l1",
        solver="saga",
        random_state=randnum,
        class_weight="balanced",
        max_iter=5000,
    ).fit(X_LRtrain, Ytrainvalid)
    stop = timeit.default_timer()
    train_time = stop - start
    start2 = timeit.default_timer()

    # Get model predictions on the test data
    LRpred = LRmodel.predict(X_LRtest)
    LRprob = LRmodel.predict_proba(X_LRtest)[:, 1]

    # get output metrics for test data
    eval_metrics_out_nothres_nocal = utils.eval_metrics(
        y_pred=LRpred,
        y_proba=LRprob,
        Y_test=Ytest,
        case_name="no_threshold_or_calibration",
        output_dir=output_dir,
        save_name=save_name,
        randnum=randnum,
    )

    # Calibrate the model
    calibrated_model = _fit_calibrator_prefit(LRmodel, X_LRtrain, Ytrainvalid)
    LRprob_calibrated = calibrated_model.predict_proba(X_LRtest)[:, 1]

    LRpred_thres, best_threshold, best_f1_score = utils.threshold_func(
        Y_test=Ytest,
        y_proba=LRprob_calibrated,
        output_dir=output_dir,
        save_name=save_name,
        randnum=randnum,
    )

    eval_metrics_out_thres_cal = utils.eval_metrics(
        y_pred=LRpred_thres,
        y_proba=LRprob_calibrated,
        Y_test=Ytest,
        case_name="with_threshold_and_calibration",
        output_dir=output_dir,
        save_name=save_name,
        randnum=randnum,
    )

    stop2 = timeit.default_timer()
    inf_time = stop2 - start2

    ### run_feature_importance is returning the covariates in this case

    if run_feature_importance:
        try:
            from lce import explr
        except ModuleNotFoundError:
            import explr

        explr.explain_func_LR(
            X=X_LRtrain,
            learn=LRmodel,
            output_dir=output_dir,
            save_name=save_name,
            Y_test=Ytest,
            randnum=randnum,
            X_test=Xtest,
            XLRtest=X_LRtest,
            n_people=np.shape(Xtrainvalid)[0],
            n_features=np.shape(Xtrainvalid)[1],
            n_time=np.shape(Xtrainvalid)[2],
            y_pred=LRpred_thres,
        )

    return (
        train_time,
        inf_time,
        eval_metrics_out_nothres_nocal,
        eval_metrics_out_thres_cal,
        best_threshold,
        best_f1_score,
    )
