# ============================================================
# 文件: main_controller.R 
# ============================================================
rm(list = ls())
gc()
setwd("C:/Users/kagirasu/works_H/ml_HULL/job/code") 

library(dplyr)
library(zoo)
library(ggplot2)

source("code_section3/proj_svm/svm_modulized_handmade/algorithm_handmade.r")
source("code_section4/RF_NEW/01_data_processor_rf.R")
source("code_section4/RF_NEW/03_strategy_engine_rf.R")

MODEL_TYPE <- "rf"
train_func <- NULL 

if (MODEL_TYPE == "rf") {
  source("code_section4/RF_NEW/02_model_trainer_rf_handmade.R")
  train_func <- train_rf_model
  
} else if (MODEL_TYPE == "xgboost") {
  source("code_section4/RF_NEW/02_model_trainer_xgb.R")
  train_func <- train_xgb_model
  
} else if (MODEL_TYPE == "lr") {
  source("code_section4/RF_NEW/02_model_trainer_lr.R")
  train_func <- train_lr_model
}
CONFIG <- list(
  input_file      = "code_section3/proj_svm/data/train.csv",
  output_dir      = paste0("output_", MODEL_TYPE, "_sniper"),
  seed            = 42,
  
  scale_features  = TRUE, 
  missing_thresh  = 0.20,
  train_end_id    = 7000,
  test_start_id   = 7001,
  test_end_id     = 7984,
  na_handling     = "zero",
  
  rf_params       = list(
    ntree      = 1000,
    mtry       = "sqrt",
    nodesize   = 10,
    max_depth  = 5,
    importance = TRUE
  ),

  xgb_params      = list(
    nrounds          = 1000,
    eta              = 0.01,
    max_depth        = 4,
    subsample        = 0.7,
    colsample_bytree = 0.7,
    min_child_weight = 5,
    gamma            = 0.1,
    early_stopping   = 50
  ),
  
  lr_params = list(
    alpha = 1
  ),
  
sniper_mode = list(
  active          = TRUE,
  lookback_window = 250,
  open_quantile   = 0.98,   # 维持高标准，只在最强时加杠杆
  
  hard_floor_open = 0.53,   # 狙击触发线
  hard_floor_close= 0.50,   # 趋势多空分界线
  
  # 指数增强仓位管理
  pos_base        = 0.3,    # 底仓: 平时拿 30%
  pos_boost       = 2,    # 增强: 狙击信号触发时，上2倍杠杆
  pos_bailout     = 0.0,    # 逃生: 极度看空(如 prob < 0.45)时清仓，可选
  bailout_thresh  = 0.5    # 逃生阈值
),
  
  target_vol       = 0.2,
  max_leverage     = 2.0,
  transaction_cost = 0.001 
)

set.seed(CONFIG$seed)
if(!dir.exists(CONFIG$output_dir)) dir.create(CONFIG$output_dir)

start_time <- Sys.time()
cat(sprintf(">>> [%s Sniper] 启动...\n", toupper(MODEL_TYPE)))

data_bundle <- process_data_rf(CONFIG$input_file, CONFIG)

cat("[Main] 调用训练函数...\n")
model_bundle <- train_func(data_bundle, CONFIG)

backtest_results <- run_strategy_backtest_rf(data_bundle, model_bundle, CONFIG)

end_time <- Sys.time()
cat("\n>>> 完成! 耗时:", round(difftime(end_time, start_time, units="mins"), 2), "分\n")

library(dplyr)

check_df <- data.frame(
  prob = backtest_results$results_df$pred_prob,
  ret  = backtest_results$results_df$market_forward_excess_returns
)

ic <- cor(check_df$prob, check_df$ret, method = "spearman")
cat(sprintf(">>> Rank IC (预测力): %.4f\n", ic))

check_df$group <- cut(check_df$prob, breaks = 5, labels = FALSE)
group_res <- check_df %>% 
  group_by(group) %>% 
  summarise(avg_ret = mean(ret, na.rm=TRUE), count = n())

print(group_res)