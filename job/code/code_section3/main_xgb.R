# ============================================================
# 主程序: XGBoost主控
# ============================================================
rm(list = ls())
gc()
setwd("C:/Users/kagirasu/works_H/ml_HULL/job/code/code_section3")

library(dplyr)
library(zoo)
library(ggplot2)

source("proj_svm/svm_modulized_handmade/algorithm_handmade.r")
source("XGB/01_data_processor_xgb.R")
source("XGB/02_model_trainer_xgb copy.R")
source("XGB/03_strategy_engine_xgb.R")



CONFIG <- list(
  input_file      = "proj_svm/data/train.csv",
  output_dir      = "output_xgb_fixed",

  use_lasso       = FALSE,
  lasso_params    = list(
    n_lambdas = 20,         # 扫描多少个 lambda
    k_folds   = 5           # 交叉验证折数
  ),
  missing_thresh  = 0.20,
  train_end_id    = 7000,
  test_start_id   = 7001,
  test_end_id     = 7984,
  na_handling     = "none", 

  xgb_params      = list(
    eta             = 0.01,   # 学习率
    max_depth       = 3,
    subsample       = 0.7,    
    colsample_bytree= 0.7,    
    min_child_weight= 10,     
    objective       = "binary:logistic",
    eval_metric     = "auc",  
    nrounds         = 10,   
    early_stopping  = 50      
  ),
  
  target_vol       = 0.2,
  max_leverage     = 2.0,
  trend_window     = 60,
  prob_roll_window = 60,
  prob_slope       = 10,
  transaction_cost = 0.001
)

if(!dir.exists(CONFIG$output_dir)) dir.create(CONFIG$output_dir)

# ============================================================
# 执行流水线
# ============================================================
start_time <- Sys.time()
cat(">>> [XGB Project] 启动...\n")

cat("\n>>> 阶段 I: 数据预处理\n")
data_bundle <- process_data_xgb(CONFIG$input_file, CONFIG)

cat("\n>>> 阶段 II: XGBoost 训练\n")
model_bundle <- train_xgb_model(data_bundle, CONFIG)

cat("\n>>> 阶段 III: 策略回测...\n")
backtest_results <- run_strategy_backtest_xgb(data_bundle, model_bundle, CONFIG)

end_time <- Sys.time()
cat("\n>>> [XGB Project] 完成! 耗时:", round(difftime(end_time, start_time, units="mins"), 2), "分\n")
