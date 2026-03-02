# ============================================================
# 主程序: SVM
# ============================================================
setwd("C:\\Users\\kagirasu\\works_H\\ml_HULL\\job\\code\\code_section3\\proj_svm")
rm(list = ls())
gc()

library(dplyr)
library(ggplot2)

source("svm_modulized_handmade/algorithm_handmade.R")
source("svm_modulized_handmade/01_data_processor.R")
source("svm_modulized_handmade/02_model_trainer.R")
source("svm_modulized_handmade/03_new.R")

CONFIG <- list(
  # --- 文件路径配置 ---
  input_file  = "data/train.csv",
  output_dir  = "output",

  model_type     = "svm",
  kernel_type    = "rbf",

  missing_thresh = 0.20,    # 缺失率剔除阈值 (20%)
  train_end_id   = 7000,    # 训练集截止 ID
  test_start_id  = 7001,    # 测试集开始 ID
  test_end_id    = 7984,    # 测试集截止 ID
  
  cv_folds       = 2,       # 交叉验证折数
  
  tune_grid      = list(
    # C = c(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0), 
    # C = c(0.1, 1.0, 10.0, 100.0, 1000.0),
    # C = c(10, 100, 1000, 10000, 50000),
    # C = c(0.1:1, 0.2),
    # C = seq(from = 0.01, to = 0.2, by = 0.01),
    # gamma = seq(from = 0.001, to = 0.015, by = 0.003) # RBF 模式下才会用到
     C = 1e-1,
     gamma = 1e-2
  ),
  
  # [调试模式] 固定参数
  # 如果不想每次都跑网格搜索，取消下面这行的注释，就能直接使用最佳参数
 
  fixed_params   = NULL,
  
  target_vol       = 0.20,    # 目标波动率
  max_leverage     = 2,       # 最大杠杆上限
  trend_window     = 60,      # 趋势均线窗口 (MA60)
  prob_roll_window = 60,      # 自适应中枢的回顾窗口 (60天)
  prob_slope       = 10,      # 信号敏感度
  transaction_cost = 0.001
)

if(!dir.exists(CONFIG$output_dir)) dir.create(CONFIG$output_dir)


# ============================================================
# 3. 程序流
# ============================================================

start_time <- Sys.time()
cat(">>> 任务启动时间:", format(start_time, "%Y-%m-%d %H:%M:%S"), "\n")

cat("\n", rep("=", 40), "\n", ">>> 阶段1: 数据预处理\n", rep("=", 40), "\n", sep="")
tryCatch({
  data_bundle <- process_data(CONFIG$input_file, CONFIG)
}, error = function(e) {
  stop("数据处理阶段出错: ", e$message)
})

cat("\n", rep("=", 40), "\n", ">>> 阶段2: 模型训练\n", rep("=", 40), "\n", sep="")
tryCatch({
  model_bundle <- train_model_entry(data_bundle, CONFIG)
}, error = function(e) {
  stop("模型训练阶段出错: ", e$message)
})

cat("\n", rep("=", 40), "\n", ">>> 阶段3: 策略回测\n", rep("=", 40), "\n", sep="")
tryCatch({
  backtest_results <- run_strategy_backtest(data_bundle, model_bundle, CONFIG)
}, error = function(e) {
  stop("策略回测阶段出错: ", e$message)
})

# ============================================================
# 4. 结束摘要
# ============================================================
end_time <- Sys.time()
duration <- round(difftime(end_time, start_time, units = "mins"), 2)

cat("\n", rep("=", 40), "\n", ">>> 所有任务执行完毕！\n", rep("=", 40), "\n", sep="")
cat("  - 总耗时:", duration, "分钟\n")
cat("  - 输出目录:", CONFIG$output_dir, "\n")
cat("  - 最终夏普比率:", round(backtest_results$metrics$Strategy[2], 4), "\n")
cat("  - 结果图表: backtest_equity_curve.png\n")