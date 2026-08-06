"""Script to run the hyperparameter search and rerun with fitted parameters for each model"""

import time
from pathlib import Path

try:
    from lce.utils import create_filename
except ModuleNotFoundError:
    from utils import create_filename


def all_run(
    data_dir: Path,
    output_dir: Path,
    device,
    data_name,
    model_name,
    X_trainvalid,
    Y_trainvalid,
    X_test,
    Y_test,
    stoc,
    randnum_train,
    randnum_split,
    randnum_stoc,
    epochs=10,
    num_optuna_trials=100,
    run_hyper_opt=False,
    run_feature_importance=False,
    folds=5,
    n_people=None,  # ✅ FIX: N 크기를 파일명에 포함시켜 N=5000/10000 결과 구분
):
    """Run hyperparameter search on train/valid, then run on train/test with chosen parameters.

    Args:
        data_dir (Path): Gives the location of the data
        output_dir (Path): Gives the location of the output
        device (int): For cuda, is 0 or 1 to give the device to run
        data_name (string): Name of the simulated data.
        model_name (string): Name of DL model; ResNet, LSTMAttention, MLSTMFCN, InceptionTime and LR
        X_trainvalid (nparray): Training input data.
        Y_trainvalid (nparray): Training output data.
        X_test (nparray): Testing input data.
        Y_test (nparray): Testing output data.
        stoc (float): The proportion of noise in the data, between 0 and 1
        randnum_train (int): Seed used to control optimisation process and weight initialisation
        randnum_split (int): Seed used to split the data into test and train groups
        randnum_stoc (int): Seed used to control which data is switched when noisy
        epochs (int, optional): Number of epochs. Defaults to 10.
        num_optuna_trials (int, optional): Number of optimisation trials. Default is 100.
        run_hypter_opt (bool, optional): If hyperparameter optimisation will be conducted. Defaults to False.
        run_feature_importance (bool, optional): Whether feature importance will be conducted. Defaults to False.
        folds (int, optional): Number of folds in k-fold cross-validation. Defaults to 5.

    Returns:
        table: Table containing model run performance and parameter details
    """

    # Record time
    timestr = time.strftime("%Y%m%d-%H%M%S")

    # Giving the filepath for the output
    # ✅ FIX: n_people을 파일명에 포함 → N=500/1000/.../10000 결과가 겹치지 않음
    n_suffix = f"_n{n_people}" if n_people is not None else ""
    save_name = create_filename(
        "model_output",
        data_name + n_suffix,
        model_name,
        stoc=int(stoc * 100),
        randsp=int(randnum_split),
        randtr=int(randnum_train),
        hype=run_hyper_opt,
        time=timestr,
    )
    output_file = (output_dir / save_name).with_suffix(".csv")

    if model_name == "LR":
        try:
            from lce import logistic_regression_model
        except ModuleNotFoundError:
            import logistic_regression_model

        output = logistic_regression_model.run_logistic_regression(
            model_name=model_name,
            output_dir=output_dir,
            output_file=output_file,
            data_name=data_name,
            stoc=stoc,
            X_trainvalid=X_trainvalid,
            Y_trainvalid=Y_trainvalid,
            X_test=X_test,
            Y_test=Y_test,
            run_hyper_opt=run_hyper_opt,
            run_feature_importance=run_feature_importance,
            epochs=epochs,
            num_optuna_trials=num_optuna_trials,
            randnum_split=randnum_split,
            randnum_train=randnum_train,
            randnum_stoc=randnum_stoc,
            folds=folds,
            device=device,
            save_name=save_name,
            metrics=None,
            timestr=timestr,
        )

    elif model_name == "XGBoost":
        try:
            from lce import xgboost_model
        except ModuleNotFoundError:
            import xgboost_model

        output = xgboost_model.run_xgboost(
            model_name=model_name,
            output_dir=output_dir,
            output_file=output_file,
            data_name=data_name,
            stoc=stoc,
            X_trainvalid=X_trainvalid,
            Y_trainvalid=Y_trainvalid,
            X_test=X_test,
            Y_test=Y_test,
            run_hyper_opt=run_hyper_opt,
            run_feature_importance=run_feature_importance,
            epochs=epochs,
            num_optuna_trials=num_optuna_trials,
            randnum_split=randnum_split,
            randnum_train=randnum_train,
            randnum_stoc=randnum_stoc,
            folds=folds,
            device=device,
            save_name=save_name,
            metrics=None,
            timestr=timestr,
        )

    else:
        # Delay fastai imports so LR/XGBoost runs do not require DL dependencies.
        from fastai.metrics import APScoreBinary, BrierScore, F1Score, RocAucBinary, accuracy

        try:
            from lce import dl_models
        except ModuleNotFoundError:
            import dl_models

        # Evaluation metrics output when fitting the DL model
        metrics = [accuracy, F1Score(), RocAucBinary(), BrierScore(), APScoreBinary()]
        output = dl_models.run_dl_models(
            model_name=model_name,
            output_dir=output_dir,
            output_file=output_file,
            data_name=data_name,
            stoc=stoc,
            X_trainvalid=X_trainvalid,
            Y_trainvalid=Y_trainvalid,
            X_test=X_test,
            Y_test=Y_test,
            run_hyper_opt=run_hyper_opt,
            run_feature_importance=run_feature_importance,
            epochs=epochs,
            num_optuna_trials=num_optuna_trials,
            randnum_split=randnum_split,
            randnum_train=randnum_train,
            randnum_stoc=randnum_stoc,
            folds=folds,
            device=device,
            save_name=save_name,
            metrics=metrics,
            timestr=timestr,
        )

    return output
