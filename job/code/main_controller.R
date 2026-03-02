rm(list = ls()); gc()
setwd("C:/Users/kagirasu/works_H/ml_HULL/job/code") 

library(dplyr)
library(zoo)
library(ggplot2)
source("code_section3/proj_svm/svm_modulized_handmade/algorithm_handmade.r")
source("code_section4/RF_NEW/01_data_processor_rf.R")
source("code_section4/RF_NEW/02_model_trainer_rf_handmade.R")
source("code_section4/RF_NEW/02_model_trainer_xgb.R")
source("code_section4/RF_NEW/03_strategy_engine_dual.R")

CONFIG <- list(
  input_file      = "code_section3/proj_svm/data/train.csv",
  output_dir      = "output_hybrid_moe", 
  seed            = 42,
  
  scale_features  = FALSE, 
  missing_thresh  = 0.20,
  train_end_id    = 7000,
  test_start_id   = 7001,
  test_end_id     = 7984,
  na_handling     = "zero",
  
  xgb_params      = list(
    nrounds=1000, eta=0.01, max_depth=4, subsample=0.7, 
    colsample_bytree=0.7, min_child_weight=5, gamma=0.1, early_stopping=50,
    seed=42
  ),
  
  rf_params       = list(
    ntree=1000, mtry="sqrt", nodesize=10, max_depth=5, importance=TRUE
  ),

  sniper_mode = list(
    active          = TRUE,
    trend_window    = 90,      
    trend_threshold = 0.52,    
    lookback_window = 250,
    open_quantile   = 0.99,
    hard_floor_open = 0.55,
    pos_base        = 0.8,     
    pos_boost       = 1.5,     
    pos_bailout     = 0.0,
    bailout_thresh  = 0.45
  )
)

set.seed(CONFIG$seed)
if(!dir.exists(CONFIG$output_dir)) dir.create(CONFIG$output_dir)

cat(">>> [Hybrid MoE] 系统启动...\n")
t_start <- Sys.time()

cat(">>> [Step 1/4] 处理数据...\n")
data_bundle <- process_data_rf(CONFIG$input_file, CONFIG)

cat(">>> [Step 2/4] 训练XGBoost...\n")
# 注意: 这里调用的是 train_xgb_model
bundle_trend <- train_xgb_model(data_bundle, CONFIG)

cat(">>> [Step 3/4] 训练Random Forest...\n")
# 注意: 这里调用的是 train_rf_model
bundle_sniper <- train_rf_model(data_bundle, CONFIG)

cat(">>> [Step 4/4] 执行混合策略回测...\n")
backtest_results <- run_strategy_dual(data_bundle, bundle_trend, bundle_sniper, CONFIG)

t_end <- Sys.time()
cat(sprintf("\n>>> 全部完成! 总耗时: %.2f 分钟\n", difftime(t_end, t_start, units="mins")))
print(backtest_results$metrics)