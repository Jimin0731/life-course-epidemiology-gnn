## Sheet for simulating data using DAG for multiple N sizes

# 1. 필수 파일 및 라이브러리 로드
# (주의: 이 파일이 같은 폴더에 있어야 합니다)
if(!file.exists("LCP_rule_functions.R")) {
  stop("LCP_rule_functions.R 파일이 없습니다. 작업 디렉토리를 확인해주세요.")
}
source("LCP_rule_functions.R")

library(simcausal)
library(reticulate)

# 2. Python numpy 설정
# (이미 환경 설정이 되어 있다고 가정합니다. 만약 오류가 나면 주석을 해제하여 환경을 잡으세요)
# use_virtualenv("r-reticulate")
np <- import("numpy") 

# 3. 공통 변수 설정
num_var <- 9 
n_time_out <- 40 
n_time <- n_time_out - 1

# 4. DAG 정의 (구조는 모든 N에 대해 동일하므로 루프 밖에서 한 번만 정의)
dag <- DAG.empty() +
  node("sex", distr = "rcat.b1", prob = c(0.5, 0.5), replaceNAw0 = TRUE) + 
  node("ethnicity", distr = "rcat.b1", prob = c(0.9, 0.1), replaceNAw0 = TRUE) + 
  node("marriage", distr = "rcat.b1", prob = c(0.75, 0.25), replaceNAw0 = TRUE) + 
  node("education", distr = "rcat.b1", prob = c(0.35, 0.65), replaceNAw0 = TRUE) + 
  node("class", distr = "rcat.b1", prob = c(0.81, 0.19), replaceNAw0 = TRUE) + 
  node("admissions", t = 0:n_time, distr = "rbern", prob = (1 / (5 * log(t + 3)) - 0.015), replaceNAw0 = TRUE) + 
  node("admissions2", t = 0:n_time, distr = "rbern", prob = (0.04 * ((class - 1) * 0.28 + 1) * ((education - 1) * 0.22 + 1)), replaceNAw0 = TRUE) + 
  node("admissions3", t = 0, distr = "rbern", prob = ((0.02 + sex / 100) * ((ethnicity - 1) * 0.17 + 1) + plogis(-5)), replaceNAw0 = TRUE) + 
  node("admissions3", t = 1:n_time, distr = "rbern", prob = ((0.02 + sex / 100) * ((ethnicity - 1) * 0.17 + 1) + plogis(-5 + admissions3[t - 1])), replaceNAw0 = TRUE) + 
  node("admissions4", t = 0, distr = "rbern", prob = plogis(-2 + admissions2[t] + admissions[t]), replaceNAw0 = TRUE) + 
  node("admissions4", t = 1, distr = "rbern", prob = plogis(-2 + ((admissions2[t] + admissions2[t - 1]) > 0) + ((admissions[t] + admissions[t - 1]) > 0)), replaceNAw0 = TRUE) + 
  node("admissions4", t = 2, distr = "rbern", prob = plogis(-2 + ((admissions2[t] + admissions2[t - 1] + admissions2[t - 2]) > 1) + ((admissions[t] + admissions[t - 1] + admissions[t - 2]) > 1)), replaceNAw0 = TRUE) +
  node("admissions4", t = 3:n_time, distr = "rbern", prob = plogis(-2 + ((admissions2[t] + admissions2[t - 1] + admissions2[t - 2] + admissions2[t - 3]) > 2) + ((admissions[t] + admissions[t - 1] + admissions[t - 2] + admissions[t - 3]) > 2)), replaceNAw0 = TRUE) +
  node("Y", t = 0:(n_time - 1), distr = "rbern", prob = plogis(-2), EFU = FALSE, replaceNAw0 = TRUE) +
  node("Y", t = n_time, distr = "rbern", prob = plogis(-1), EFU = FALSE, replaceNAw0 = TRUE)
dag <- set.DAG(dag)

# 5. 함수 및 파일 이름 매핑
function_names <- c(
  "repeat3", "repeat4", "repeat2_2",
  "order2", "order3", "order4",
  "timing2_0", "timing3_2", "timing4_2", "timing4_4",
  "critical30", "sensitive4", "sensitive5", "weighted4"
)

paper_names <- c(
  "Repeats1", "Repeats2", "Repeats3",
  "Order1", "Order2", "Order3",
  "Timing1", "Timing2", "Timing3", "Timing4",
  "Period1", "Period2", "Period3", "Period4"
)

# =========================================================
# 6. 메인 루프: N 사이즈별 데이터 생성 및 저장
# =========================================================
n_sizes_list <- c(500, 1000, 2000, 5000, 10000)

for (current_n in n_sizes_list) {
  
  # 폴더 이름 설정 (예: data_500)
  data_dir <- paste0("data_", current_n)
  
  # 폴더가 없으면 생성
  if (!file.exists(data_dir)){
    dir.create(data_dir)
    cat(paste("\n[INFO] 폴더 생성됨:", data_dir, "\n"))
  }
  
  cat(paste(">>> 처리 중: N =", current_n, "\n"))
  
  # 시드 설정 (재현성을 위해)
  set.seed(9)
  
  # 데이터 시뮬레이션
  dat_long <- sim(dag, n = current_n)
  
  # 데이터 전처리 (Array 변환)
  data <- dat_long[-1]
  data_out <- array(dim = c(dim(data)[1], num_var, n_time + 1))
  y_out <- array(dim = c(dim(data)[1], n_time + 1))
  
  # 변수 매핑
  data_out[, 1, ] <- array(replicate(40, unlist(data[1])), dim = c(dim(data)[1], n_time + 1)) - 1
  data_out[, 2, ] <- array(replicate(40, unlist(data[2])), dim = c(dim(data)[1], n_time + 1)) - 1
  data_out[, 3, ] <- array(replicate(40, unlist(data[3])), dim = c(dim(data)[1], n_time + 1)) - 1
  data_out[, 4, ] <- array(replicate(40, unlist(data[4])), dim = c(dim(data)[1], n_time + 1)) - 1
  data_out[, 5, ] <- array(replicate(40, unlist(data[5])), dim = c(dim(data)[1], n_time + 1)) - 1
  data_out[, 6, ] <- array(unlist(data[seq(from = 6, by = 5, to = dim(data)[2])]), dim = c(dim(data)[1], n_time + 1))
  data_out[, 7, ] <- array(unlist(data[seq(from = 7, by = 5, to = dim(data)[2])]), dim = c(dim(data)[1], n_time + 1))
  data_out[, 8, ] <- array(unlist(data[seq(from = 8, by = 5, to = dim(data)[2])]), dim = c(dim(data)[1], n_time + 1))
  data_out[, 9, ] <- array(unlist(data[seq(from = 9, by = 5, to = dim(data)[2])]), dim = c(dim(data)[1], n_time + 1))
  
  y_out <- array(unlist(data[seq(from = 10, by = 5, to = dim(data)[2])]), dim = c(dim(data)[1], n_time + 1))
  x_out <- data_out
  
  # X 데이터 저장 (모든 규칙에 공통)
  np$save(paste0(data_dir, "/data_X.npy"), r_to_py(x_out))
  
  # 각 규칙(Rule)별로 Y 데이터 생성 및 저장
  for (i in seq_along(function_names)) {
    # 함수 실행
    y_outcheck <- eval(parse(text = (paste(function_names[i], "(x_out,y_out)", sep = ""))))
    
    # 파일명 생성 및 저장 (예: data_Repeats1_Y.npy)
    name <- paste("data_", paper_names[i], sep = "")
    np$save(paste0(data_dir, "/", name, "_Y.npy"), r_to_py(y_outcheck))
  }
  
  cat(paste("    완료: ", data_dir, "에 파일 저장됨.\n"))
}

cat("\n 모든 작업이 완료되었습니다! \n")