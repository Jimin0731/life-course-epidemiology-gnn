## Sheet for simulating data using DAG

# Source the functions file
source("LCP_rule_functions.R")

# Load libraries
library(simcausal)
library(reticulate)

# Define variables for simulating data
num_var <- 9 # number of variables in model
n_people <- 20000#100000 # number of people in model
n_time_out <- 40 #number of time points in model
n_time <- n_time_out - 1
set.seed(9)

# Creates DAG
dag <- DAG.empty() +
  node("sex", distr = "rcat.b1", prob = c(0.5, 0.5), replaceNAw0 = TRUE) + ## fem/mal
  node("ethnicity", distr = "rcat.b1", prob = c(0.9, 0.1), replaceNAw0 = TRUE) + ## white/other of mother
  node("marriage", distr = "rcat.b1", prob = c(0.75, 0.25), replaceNAw0 = TRUE) + ## married/single of mother
  node("education", distr = "rcat.b1", prob = c(0.35, 0.65), replaceNAw0 = TRUE) + ## degree or a level/gcse of mother
  node("class", distr = "rcat.b1", prob = c(0.81, 0.19), replaceNAw0 = TRUE) + ## nonmanual/manual of mother
  node("admissions", t = 0:n_time, distr = "rbern", prob = (1 / (5 * log(t + 3)) - 0.015), replaceNAw0 = TRUE) + ## general admissions
  node("admissions2", t = 0:n_time, distr = "rbern", prob = (0.04 * ((class - 1) * 0.28 + 1) * ((education - 1) * 0.22 + 1)), replaceNAw0 = TRUE) + ## something different affected by variables
  node("admissions3", t = 0, distr = "rbern", prob = ((0.02 + sex / 100) * ((ethnicity - 1) * 0.17 + 1) + plogis(-5)), replaceNAw0 = TRUE) + ## general admissions
  node("admissions3", t = 1:n_time, distr = "rbern", prob = ((0.02 + sex / 100) * ((ethnicity - 1) * 0.17 + 1) + plogis(-5 + admissions3[t - 1])), replaceNAw0 = TRUE) + ## general admissions
  node("admissions4", t = 0, distr = "rbern", prob = plogis(-2 + admissions2[t] + admissions[t]), replaceNAw0 = TRUE) + ## something different affected by variables
  node("admissions4", t = 1, distr = "rbern", prob = plogis(-2 + ((admissions2[t] + admissions2[t - 1]) > 0) + ((admissions[t] + admissions[t - 1]) > 0)), replaceNAw0 = TRUE) + ## something different affected by variables
  node("admissions4", t = 2, distr = "rbern", prob = plogis(-2 + ((admissions2[t] + admissions2[t - 1] + admissions2[t - 2]) > 1) + ((admissions[t] + admissions[t - 1] + admissions[t - 2]) > 1)), replaceNAw0 = TRUE) +
  node("admissions4", t = 3:n_time, distr = "rbern", prob = plogis(-2 + ((admissions2[t] + admissions2[t - 1] + admissions2[t - 2] + admissions2[t - 3]) > 2) + ((admissions[t] + admissions[t - 1] + admissions[t - 2] + admissions[t - 3]) > 2)), replaceNAw0 = TRUE) +
  node("Y", t = 0:(n_time - 1), distr = "rbern", prob = plogis(-2), EFU = FALSE, replaceNAw0 = TRUE) +
  node("Y", t = n_time, distr = "rbern", prob = plogis(-1), EFU = FALSE, replaceNAw0 = TRUE)
dag <- set.DAG(dag)

# ==============================================================================
# [NEW] Export DAG Structure for Python GNN
# ==============================================================================

# 1. define names of 9 nodes
node_names <- c(
  "sex",          # 1
  "ethnicity",    # 2
  "marriage",     # 3
  "education",    # 4
  "class",        # 5
  "admissions",   # 6
  "admissions2",  # 7
  "admissions3",  # 8
  "admissions4"   # 9
)

# 2. define parents relationship 
parents <- list(
  sex         = character(0),
  ethnicity   = character(0),
  marriage    = character(0),
  education   = character(0),
  class       = character(0),
  admissions  = character(0),          
  admissions2 = c("class", "education"),
  admissions3 = c("sex", "ethnicity"),
  admissions4 = c("admissions2", "admissions")
)

# 3. generate edge list (Source -> Destination)
edges <- do.call(rbind, lapply(names(parents), function(child){
  ps <- parents[[child]]
  if (length(ps) == 0) return(NULL)
  cbind(src = ps, dst = rep(child, length(ps)))
}))

# 4. file save
if (!dir.exists("data/")) dir.create("data/") 


export_dir <- "data/" 

write.csv(edges, paste0(export_dir, "dag_edges.csv"), row.names = FALSE, quote = FALSE)
writeLines(node_names, paste0(export_dir, "dag_node_names.txt"))

cat("SUCCESS: Saved DAG edges to", paste0(export_dir, "dag_edges.csv"), "\n")


# 1. set scenarios
n_sizes <- c(500, 1000, 2000, 5000, 10000, 20000)  # amount of data

# 2. rules 
function_names <- c(
  "order1", "order2", "order3", 
  "repeat2", "repeat3", "repeat4",
  "timing1_0", "timing2_0", "timing3_2",
  "critical30", "sensitive4", "weighted4"
)

paper_names <- function_names 

library(reticulate)
use_virtualenv("r-reticulate") 
np <- import("numpy")

for (n_p in n_sizes) {
  cat("\n========================================\n")
  cat(" Generating Data for N =", n_p, "\n")
  cat("========================================\n")
  
  # (1) Generate folder (ea. data_1000, data_2000)
  current_dir <- paste0("data_", n_p, "/")
  if (!dir.exists(current_dir)) dir.create(current_dir)
  
  # (2) Simulation
  dat_long <- sim(dag, n = n_p)
  data <- dat_long[-1]
  
  # (3) Data array change (N x 9 x 40)
  data_out <- array(dim = c(n_p, num_var, n_time + 1))
  y_out_raw <- array(unlist(data[seq(from = 10, by = 5, to = dim(data)[2])]), dim = c(n_p, n_time + 1))
  
  data_out[, 1, ] <- array(replicate(40, unlist(data[1])), dim = c(n_p, n_time + 1)) - 1
  data_out[, 2, ] <- array(replicate(40, unlist(data[2])), dim = c(n_p, n_time + 1)) - 1
  data_out[, 3, ] <- array(replicate(40, unlist(data[3])), dim = c(n_p, n_time + 1)) - 1
  data_out[, 4, ] <- array(replicate(40, unlist(data[4])), dim = c(n_p, n_time + 1)) - 1
  data_out[, 5, ] <- array(replicate(40, unlist(data[5])), dim = c(n_p, n_time + 1)) - 1
  data_out[, 6, ] <- array(unlist(data[seq(from = 6, by = 5, to = dim(data)[2])]), dim = c(n_p, n_time + 1))
  data_out[, 7, ] <- array(unlist(data[seq(from = 7, by = 5, to = dim(data)[2])]), dim = c(n_p, n_time + 1))
  data_out[, 8, ] <- array(unlist(data[seq(from = 8, by = 5, to = dim(data)[2])]), dim = c(n_p, n_time + 1))
  data_out[, 9, ] <- array(unlist(data[seq(from = 9, by = 5, to = dim(data)[2])]), dim = c(n_p, n_time + 1))
  
  x_out <- data_out
  
  # (4) X data save
  np$save(paste0(current_dir, "data_X.npy"), r_to_py(x_out))
  cat("  -> Saved X data to:", current_dir, "\n")
  
  # (5) Y data generate and save (iterate by rules)
  for (i in seq_along(function_names)) {
    fname <- function_names[i]
    pname <- paper_names[i]
    
    # implement function
    set.seed(9)
    y_generated <- eval(parse(text = paste0(fname, "(x_out, y_out_raw)")))
    
    # Y save
    np$save(paste0(current_dir, "data_", pname, "_Y.npy"), r_to_py(y_generated))
  }
  cat("  -> Saved all Y rules for N =", n_p, "\n")
}