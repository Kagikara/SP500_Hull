# ============================================================
# 主程序: Logistic 回归量化策略主控 (完整版)
# 文件路径: main_logistic.R
# ============================================================

rm(list = ls())
gc()
# [请根据实际情况修改工作目录]
setwd("C:/Users/kagirasu/works_H/ml_HULL/job/code/code_section3")

library(dplyr)
library(zoo)
library(ggplot2)

# 加载模块
source("proj_svm/svm_modulized_handmade/algorithm_handmade.R")
source("logistic/01_data_processor_logistic.R")
source("logistic/02_model_trainer_logistic.R")
source("logistic/03_strategy_engine_logistic.R") # <--- 新增
source("logistic/04_visualization_logistic.R")

# ============================================================
# 全局配置
# ============================================================
CONFIG <- list(
  input_file      = "proj_svm/data/train.csv",
  output_dir      = "output_logistic",
  
  # --- 数据切分参数 ---
  train_end_id    = 7000,   # 训练集截止
  test_start_id   = 7001,   # 测试集开始
  test_end_id     = 7984,   # 测试集截止
  
  # --- 特征列索引 ---
  p_cols          = 58:70,  # P_star 原始列
  v_cols          = 83:95,  # V_star 原始列
  
  # --- 模型参数 ---
  use_interaction = TRUE,   # 是否使用 P*V 交互项
  
  # --- 策略回测参数 ---
  target_vol       = 0.2,    # 目标波动率 15%
  max_leverage     = 2.0,     # 最大杠杆
  transaction_cost = 0.001,   # 交易成本 (万分之十)
  
  # 调试选项
  save_plots       = TRUE
)

# 创建输出目录
if(!dir.exists(CONFIG$output_dir)) dir.create(CONFIG$output_dir)

# ============================================================
# 执行流水线
# ============================================================
start_time <- Sys.time()
cat(">>> [Logit Project] 任务启动...\n")

# 1. 数据处理
# ------------------------------------------------------------
tryCatch({
  data_bundle <- process_data_logistic(CONFIG$input_file, CONFIG)
}, error = function(e) stop("数据处理失败: ", e$message))

# 2. 模型训练
# ------------------------------------------------------------
tryCatch({
  model_results <- train_logistic_model(data_bundle, CONFIG)
  # 保存模型对象
  saveRDS(model_results$model, file.path(CONFIG$output_dir, "logistic_model.rds"))
  visualize_logistic_results(data_bundle, model_results, CONFIG)
}, error = function(e) stop("模型训练失败: ", e$message))

# 3. 策略回测 (新增)
# ------------------------------------------------------------
tryCatch({
  backtest_results <- run_logistic_backtest(data_bundle, model_results, CONFIG)
}, error = function(e) stop("策略回测失败: ", e$message))

# ============================================================
# 4. 结束摘要
# ============================================================
end_time <- Sys.time()
duration <- round(difftime(end_time, start_time, units = "mins"), 2)

cat("\n", rep("=", 40), "\n", ">>> [Logit Project] 执行完毕！\n", rep("=", 40), "\n", sep="")
cat("  - 总耗时:", duration, "分钟\n")

if(exists("backtest_results") && !is.null(backtest_results$metrics)) {
  sharpe <- backtest_results$metrics[backtest_results$metrics$Metric == "Sharpe", "Strategy"]
  cat("  - 最终费后夏普比率:", sharpe, "\n")
}

cat("  - 结果已保存至:", CONFIG$output_dir, "\n")