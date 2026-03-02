# ============================================================
# 模块 1: XGB数据处理器
# ============================================================

library(dplyr)
library(tidyr)
library(zoo)

process_data_xgb <- function(file_path, config) {
  
  cat("[Data-XGB] 读取数据:", file_path, "...\n")
  if (!file.exists(file_path)) stop("找不到文件")
  raw_data <- read.csv(file_path, stringsAsFactors = FALSE)
  
  target_col_raw <- "market_forward_excess_returns"
  id_cols <- c("date_id", "forward_returns", "risk_free_rate")
  potential_features <- setdiff(names(raw_data), c(target_col_raw, id_cols))

  data_period <- raw_data %>% filter(date_id >= 1006) %>% arrange(date_id)
  
  # ------------------------------------------------------------
  # 特征筛选防泄露
  # ------------------------------------------------------------
  cat("[Data-XGB] 基于训练集统计缺失率...\n")
  train_subset_stats <- data_period %>% filter(date_id <= config$train_end_id)
  
  if(nrow(train_subset_stats) == 0) stop("错误: 训练集为空")
  
  missing_ratio <- colMeans(is.na(train_subset_stats[potential_features]))
  cols_to_drop <- names(missing_ratio[missing_ratio > config$missing_thresh])
  final_feature_cols <- setdiff(potential_features, cols_to_drop)
  
  cat("  - 剔除特征:", length(cols_to_drop), "\n")
  cat("  - 保留特征:", length(final_feature_cols), "\n")
  
  data_selected <- data_period %>% 
    select(date_id, all_of(target_col_raw), all_of(final_feature_cols))
  
  complete_data <- data_selected[!is.na(data_selected[[target_col_raw]]), ]
  
  complete_data$y_target <- ifelse(complete_data[[target_col_raw]] >= 0, 1, 0)
  
  train_set <- complete_data %>% filter(date_id <= config$train_end_id)
  test_set  <- complete_data %>% filter(date_id >= config$test_start_id & date_id <= config$test_end_id)
  
  return(list(
    X_train = train_set[final_feature_cols],
    y_train = train_set$y_target,
    X_test  = test_set[final_feature_cols],
    y_test  = test_set$y_target,
    raw_train_set = train_set,
    raw_test_set  = test_set,
    feature_cols  = final_feature_cols
  ))
}