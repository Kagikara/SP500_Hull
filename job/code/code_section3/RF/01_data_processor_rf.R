# ============================================================
# 模块 1: RF数据处理器
# ============================================================

library(dplyr)
library(tidyr)
library(zoo)

#' @param file_path 原始csv文件路径
#' @param config 全局配置列表
#' @return list 包含处理好的训练集、测试集
process_data_rf <- function(file_path, config) {
  
  cat("[Data-RF] 正在读取数据:", file_path, "...\n")
  
  if (!file.exists(file_path)) stop("错误: 找不到数据文件 -> ", file_path)
  raw_data <- read.csv(file_path, stringsAsFactors = FALSE)
  
  target_col_raw <- "market_forward_excess_returns"
  id_cols <- c("date_id", "forward_returns", "risk_free_rate")
  potential_features <- setdiff(names(raw_data), c(target_col_raw, id_cols))
  
  data_period <- raw_data %>%
    filter(date_id >= 1006) %>%
    arrange(date_id)
  
  # ------------------------------------------------------------
  cat("[Data-RF] 正在基于训练集 (date_id <=", config$train_end_id, ") 计算缺失率...\n")
  
  train_subset_for_stats <- data_period %>% 
    filter(date_id <= config$train_end_id)
  
  if(nrow(train_subset_for_stats) == 0) stop("错误: 训练集为空，请检查 train_end_id 配置")
  
  missing_ratio <- colMeans(is.na(train_subset_for_stats[potential_features]))
  cols_to_drop <- names(missing_ratio[missing_ratio > config$missing_thresh])
  final_feature_cols <- setdiff(potential_features, cols_to_drop)
  # ------------------------------------------------------------
  
  cat("[Data-RF] 特征筛选:\n")
  cat("  - 剔除高缺失特征:", length(cols_to_drop), "\n")
  cat("  - 最终保留特征:", length(final_feature_cols), "\n")

  data_selected <- data_period %>% 
    select(date_id, all_of(target_col_raw), all_of(final_feature_cols))
  
  data_filled <- na.locf(data_selected, na.rm = FALSE)
  data_filled[final_feature_cols][is.na(data_filled[final_feature_cols])] <- 0
  complete_data <- data_filled[!is.na(data_filled[[target_col_raw]]), ]
  cat("[Data-RF] 缺失值处理完成。\n")
  
  complete_data$y_target <- factor(
    ifelse(complete_data[[target_col_raw]] >= 0, 1, 0), 
    levels = c(0, 1)
  )
  
  train_set <- complete_data %>% filter(date_id <= config$train_end_id)
  test_set  <- complete_data %>% filter(date_id >= config$test_start_id & date_id <= config$test_end_id)
  
  X_train <- train_set[final_feature_cols]
  X_test  <- test_set[final_feature_cols]
  
  return(list(
    X_train = X_train,
    y_train = train_set$y_target,
    X_test  = X_test,
    y_test  = test_set$y_target,
    raw_train_set = train_set,
    raw_test_set  = test_set,
    feature_cols  = final_feature_cols
  ))
}