"""
메인 실행 스크립트 - LR, XGBoost, ResNet을 N=500~10000으로 실행
사용법:
    cd /Users/jiminbyun/Risk_Factor_simulation_DL/src/lce
    python Main_run_baseline.py
"""

import importlib.util
import sys
import traceback
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# ==============================================================================
# 설정
# ==============================================================================

# ✅ 데이터가 저장된 폴더 경로
DATA_DIR = Path("/Users/jiminbyun/Risk_Factor_simulation_DL/Data_simulation")

# ✅ 결과를 저장할 폴더 경로
OUTPUT_DIR = Path("/Users/jiminbyun/Risk_Factor_simulation_DL/output/model_results")

# ✅ 실험할 N 크기
N_SIZES = [20000]

# ✅ paper_name → 실제 Y 파일명 매핑
#    data_{N}/data_{FILE_KEY}_Y.npy 형태로 저장되어 있음
RULE_FILE_MAP = {
    "Order1": "order1",
    "Order2": "order2",
    "Order3": "order3",
    "Repeats1": "repeat2",
    "Repeats2": "repeat3",
    "Repeats3": "repeat4",
    "Timing1": "timing1_0",
    "Timing2": "timing2_0",
    "Timing3": "timing3_2",
    "Timing4": "timing4_4",  # R 재생성 후 사용 가능
    "Period1": "critical30",
    "Period2": "sensitive4",
    "Period3": "sensitive5",  # R 재생성 후 사용 가능
    "Period4": "weighted4",
}

# ✅ 실험할 모델
MODELS = ["LR", "XGBoost", "ResNet"]

# 모델별 필수 라이브러리
MODEL_DEPENDENCIES = {
    "LR": ["optuna", "sklearn"],
    "XGBoost": ["xgboost", "sklearn"],
    "ResNet": ["torch", "fastai", "tsai", "IPython"],
}

# ✅ 실험 설정
STOC = 0
RANDNUM_SPLIT = 43
RANDNUM_TRAIN = 42
RANDNUM_STOC = 0
EPOCHS = 10
RUN_HYPER_OPT = False
TEST_SIZE = 0.2

# CPU/GPU 자동 선택
try:
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

# ==============================================================================
# 데이터 로드
# ==============================================================================


def load_data(data_dir: Path, n_people: int, rule: str):
    """데이터 로드 및 train/test 분할"""
    folder = data_dir / f"data_{n_people}"
    file_key = RULE_FILE_MAP[rule]

    x_path = folder / "data_X.npy"
    y_path = folder / f"data_{file_key}_Y.npy"

    if not x_path.exists():
        raise FileNotFoundError(f"X 파일 없음: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Y 파일 없음: {y_path}")

    X = np.load(x_path)  # (N, 9, 40)
    Y = np.load(y_path)  # (N, 40)

    # X/Y 크기 불일치 보정
    if X.shape[0] != Y.shape[0]:
        min_size = min(X.shape[0], Y.shape[0])
        print(f"  ⚠️  크기 불일치: X={X.shape[0]}, Y={Y.shape[0]} → {min_size}으로 조정")
        X = X[:min_size]
        Y = Y[:min_size]

    # 마지막 타임스텝의 Y 값을 레이블로 사용
    y_label = Y[:, -1].astype(int)

    # Stratified train/test split
    X_trainvalid, X_test, Y_trainvalid, Y_test = train_test_split(
        X,
        y_label,
        test_size=TEST_SIZE,
        random_state=RANDNUM_SPLIT,
        stratify=y_label if y_label.sum() >= 2 else None,
    )

    pos_rate = y_label.mean()
    return X_trainvalid, X_test, Y_trainvalid, Y_test, pos_rate


# ==============================================================================
# 메인 실행
# ==============================================================================


def main():
    # Make `lce` package imports work when executed as a script from any cwd.
    src_dir = Path(__file__).resolve().parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    try:
        from lce.run_all_models import all_run
    except ModuleNotFoundError:
        from run_all_models import all_run

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rules = list(RULE_FILE_MAP.keys())
    total = len(N_SIZES) * len(rules) * len(MODELS)
    done = 0
    success = 0
    skipped = 0
    errors = 0

    print(f"\n{'=' * 60}")
    print(f"실험 시작: {total}개 조합 (N×Rule×Model)")
    print(f"  N:      {N_SIZES}")
    print(f"  Rules:  {rules}")
    print(f"  Models: {MODELS}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'=' * 60}\n")

    for n_people in N_SIZES:
        print(f"\n{'=' * 60}")
        print(f"N = {n_people}")
        print(f"{'=' * 60}")

        for rule in rules:
            print(f"\n  [Rule: {rule}]")

            # 데이터 로드
            try:
                X_trainvalid, X_test, Y_trainvalid, Y_test, pos_rate = load_data(
                    DATA_DIR, n_people, rule
                )
                print(
                    f"    로드 완료 - shape: {X_trainvalid.shape}, 양성비율: {pos_rate:.3f}"
                )
            except FileNotFoundError:
                print("    ⚠️  파일 없음 → skip")
                skipped += len(MODELS)
                continue
            except Exception as e:
                print(f"    ❌ 데이터 로드 오류 → skip: {e}")
                skipped += len(MODELS)
                continue

            # 양성 샘플이 너무 적으면 skip
            if Y_test.sum() < 2:
                print(f"    ⚠️  테스트 양성 샘플 {Y_test.sum()}개 → 평가 불가, skip")
                skipped += len(MODELS)
                continue

            for model_name in MODELS:
                done += 1
                print(f"\n    ▶ [{done}/{total}] {model_name} 학습 중...", flush=True)

                # 필수 의존성이 없으면 즉시 skip (긴 traceback 반복 방지)
                missing = [
                    dep
                    for dep in MODEL_DEPENDENCIES.get(model_name, [])
                    if importlib.util.find_spec(dep) is None
                ]
                if missing:
                    print(f"      ⚠️  의존성 없음 → skip: {', '.join(missing)}")
                    skipped += 1
                    continue

                # 이미 결과 있으면 skip
                existing = list(
                    OUTPUT_DIR.glob(
                        f"model_output_{rule}_n{n_people}_{model_name}*.csv"
                    )
                )
                if existing:
                    print(f"      ↩️  이미 결과 존재 → skip ({existing[0].name})")
                    skipped += 1
                    continue

                try:
                    all_run(
                        data_dir=DATA_DIR,
                        output_dir=OUTPUT_DIR,
                        device=DEVICE,
                        data_name=rule,
                        model_name=model_name,
                        X_trainvalid=X_trainvalid,
                        Y_trainvalid=Y_trainvalid,
                        X_test=X_test,
                        Y_test=Y_test,
                        stoc=STOC,
                        randnum_train=RANDNUM_TRAIN,
                        randnum_split=RANDNUM_SPLIT,
                        randnum_stoc=RANDNUM_STOC,
                        epochs=EPOCHS,
                        run_hyper_opt=RUN_HYPER_OPT,
                        run_feature_importance=False,
                        folds=5,
                        n_people=n_people,
                    )
                    success += 1
                    print("      ✅ 완료")

                except Exception as e:
                    errors += 1
                    print(f"      ❌ 오류 발생: {e}")
                    traceback.print_exc()
                    continue

    print(f"\n{'=' * 60}")
    print("실험 완료!")
    print(f"  성공: {success}개")
    print(f"  Skip: {skipped}개")
    print(f"  오류: {errors}개")
    print(f"  결과 위치: {OUTPUT_DIR}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
