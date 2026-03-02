# ============================================================
# 模块 1: SVM数据预处理
# ============================================================

library(dplyr)
library(tidyr)
library(zoo)

#' 处理数据的主函数
process_data <- function(file_path, config) {
  
  cat("[Data] 正在读取数据:", file_path, "...\n")

  if (!file.exists(file_path)) stop("错误: 找不到数据文件 -> ", file_path)
  raw_data <- read.csv(file_path, stringsAsFactors = FALSE)
  
  target_col_raw <- "market_forward_excess_returns"
  id_cols <- c("date_id", "forward_returns", "risk_free_rate")
  potential_features <- setdiff(names(raw_data), c(target_col_raw, id_cols))
  
  # 数据切片: 仅处理有效时间段
  data_period <- raw_data %>%
    filter(date_id >= 1006) %>%
    arrange(date_id)
  
  # ------------------------------------------------------------
  # 特征筛选防泄露
  # 必须仅基于训练集 (date_id <= train_end_id) 来决定删除哪些列
  # ------------------------------------------------------------
  cat("[Data] 正在基于训练集统计特征缺失率...\n")
  
  train_subset_for_stats <- data_period %>% 
    filter(date_id <= config$train_end_id)
  
  if(nrow(train_subset_for_stats) == 0) stop("错误: 训练集为空，请检查配置！")
  
  missing_ratio <- colMeans(is.na(train_subset_for_stats[potential_features]))
  
  # 剔除特征
  threshold <- config$missing_thresh
  cols_to_drop <- names(missing_ratio[missing_ratio > threshold])
  final_feature_cols <- setdiff(potential_features, cols_to_drop)
  
  cat("[Data] 特征筛选报告:\n")
  cat("  - 初始特征数:", length(potential_features), "\n")
  cat("  - 剔除特征数:", length(cols_to_drop), "(基于训练集)\n")
  cat("  - 最终保留特征:", length(final_feature_cols), "\n")
  
  # ------------------------------------------------------------
  # 样本清洗与填充
  data_pruned <- data_period %>% 
    select(date_id, all_of(target_col_raw), all_of(final_feature_cols))
  
  data_filled <- na.locf(data_pruned, na.rm = FALSE)
  
  complete_data <- na.omit(data_filled)
  
  complete_data$y_target <- factor(
    ifelse(complete_data[[target_col_raw]] >= 0, 1, 0), 
    levels = c(0, 1)
  )
  
  train_set <- complete_data %>% filter(date_id <= config$train_end_id)
  test_set  <- complete_data %>% filter(date_id >= config$test_start_id & date_id <= config$test_end_id)
  
  # ------------------------------------------------------------
  # Z-Score 标准化
  # ------------------------------------------------------------
  cat("[Data] 执行 Z-Score 标准化\n")
  scaler_params <- algo_zscore_fit(train_set[final_feature_cols])
  
  X_train_mat <- algo_zscore_transform(train_set[final_feature_cols], scaler_params)
  X_test_mat  <- algo_zscore_transform(test_set[final_feature_cols], scaler_params)
  
  return(list(
    X_train = as.data.frame(X_train_mat),
    y_train = train_set$y_target,
    X_test  = as.data.frame(X_test_mat),
    y_test  = test_set$y_target,
    raw_train_set = train_set,
    raw_test_set  = test_set,
    scaling_params = scaler_params
  ))
}