# ============================================================
# 主程序: RF主控
# ============================================================

rm(list = ls())
gc()
setwd("C:/Users/kagirasu/works_H/ml_HULL/job/code/code_section3")

library(dplyr)
library(zoo)
library(ggplot2)

source("proj_svm/svm_modulized_handmade/algorithm_handmade.r")
source("RF/01_data_processor_rf.R")
source("RF/02_model_trainer_rf_handmade.R")
source("RF/03_strategy_engine_rf.r")

CONFIG <- list(
  input_file      = "proj_svm/data/train.csv",
  output_dir      = "output_rf_fixed",

  missing_thresh  = 0.20,
  train_end_id    = 7000,
  test_start_id   = 7001,
  test_end_id     = 7984,
  na_handling     = "zero",
  
  rf_params       = list(
    ntree      = 500,     # 树的数量
    mtry       = "sqrt",  # 每次分裂随机选取的特征数
    nodesize   = 5,       # 叶节点最小样本数
    max_depth  = 10,      # 限制深度防止过拟合
    importance = TRUE     # 特征重要性计算
  ),
  
  target_vol       = 0.2,     # 目标波动率
  max_leverage     = 2.0,     # 最大杠杆
  trend_window     = 60,      # 趋势均线窗口
  prob_roll_window = 60,      # 自适应中枢窗口
  prob_slope       = 8,       # 信号敏感度
  transaction_cost = 0.001    # 交易费用
)

if(!dir.exists(CONFIG$output_dir)) dir.create(CONFIG$output_dir)

# ============================================================
# 程序流
# ============================================================

start_time <- Sys.time()
cat(">>> [RF Project] 任务启动时间:", format(start_time, "%Y-%m-%d %H:%M:%S"), "\n")

cat("\n", rep("=", 40), "\n", ">>> 阶段 I: 数据预处理\n", rep("=", 40), "\n", sep="")
tryCatch({
  data_bundle <- process_data_rf(CONFIG$input_file, CONFIG)
}, error = function(e) {
  stop("数据处理阶段出错: ", e$message)
})

cat("\n", rep("=", 40), "\n", ">>> 阶段 II: 随机森林训练\n", rep("=", 40), "\n", sep="")
tryCatch({
  model_bundle <- train_rf_model(data_bundle, CONFIG)
}, error = function(e) {
  stop("模型训练阶段出错: ", e$message)
})

cat("\n", rep("=", 40), "\n", ">>> 阶段 III: 策略回测\n", rep("=", 40), "\n", sep="")
tryCatch({
  backtest_results <- run_strategy_backtest_rf(data_bundle, model_bundle, CONFIG)
}, error = function(e) {
  stop("策略回测阶段出错: ", e$message)
})

# ============================================================
# 结束摘要
# ============================================================
end_time <- Sys.time()
duration <- round(difftime(end_time, start_time, units = "mins"), 2)

cat("\n", rep("=", 40), "\n", ">>> [RF Project] 所有任务执行完毕！\n", rep("=", 40), "\n", sep="")
cat("  - 总耗时:", duration, "分钟\n")
cat("  - 输出目录:", CONFIG$output_dir, "\n")
if(!is.null(backtest_results$metrics)) {
  cat("  - 最终费后夏普比率:", backtest_results$metrics[backtest_results$metrics$Metric=="Sharpe", "Strategy"], "\n")
}
cat("  - 结果文件已保存。\n")