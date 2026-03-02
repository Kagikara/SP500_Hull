# ============================================================
# 模块 1: 通用数据处理器 (支持标准化)
# 文件路径: code_section4/RF_NEW/01_data_processor_rf.R
# ============================================================

library(dplyr)
library(tidyr)
library(zoo)

process_data_rf <- function(file_path, config) {
  
  cat("[Data] 正在读取数据:", file_path, "...\n")
  if (!file.exists(file_path)) stop("错误: 找不到数据文件")
  raw_data <- read.csv(file_path, stringsAsFactors = FALSE)
  
  # 1. 特征定义
  target_col_raw <- "market_forward_excess_returns"
  id_cols <- c("date_id", "forward_returns", "risk_free_rate")
  core_factors <- c("M4", "P4") 
  
  potential_features <- setdiff(names(raw_data), c(target_col_raw, id_cols))
  
  # 2. 切片
  data_period <- raw_data %>%
    filter(date_id >= 1006) %>%
    arrange(date_id)
  
  # 3. 缺失值筛选 (基于 Train)
  train_subset <- data_period %>% filter(date_id <= config$train_end_id)
  missing_ratio <- colMeans(is.na(train_subset[potential_features]))
  cols_to_drop <- names(missing_ratio[missing_ratio > config$missing_thresh])
  cols_to_drop <- setdiff(cols_to_drop, core_factors)
  final_features <- setdiff(potential_features, cols_to_drop)
  
  cat(sprintf("[Data] 保留特征数: %d\n", length(final_features)))
  
  # 4. 填充 NA
  data_selected <- data_period %>% 
    select(date_id, all_of(target_col_raw), all_of(final_features))
  
  # 先前向填充，再补0
  data_filled <- na.locf(data_selected, na.rm = FALSE)
  data_filled[final_features][is.na(data_filled[final_features])] <- 0
  
  complete_data <- data_filled[!is.na(data_filled[[target_col_raw]]), ]
  
  # 5. 构造 Target
  complete_data$y_target <- factor(
    ifelse(complete_data[[target_col_raw]] >= 0, 1, 0), levels = c(0, 1)
  )
  
  # 6. 切分
  train_set <- complete_data %>% filter(date_id <= config$train_end_id)
  test_set  <- complete_data %>% filter(date_id >= config$test_start_id & date_id <= config$test_end_id)
  
  X_train <- train_set[final_features]
  X_test  <- test_set[final_features]
  
  # ============================================================
  # [新增] 7. 标准化 (Z-Score Standardization)
  # 仅当配置要求时执行 (LR 需要，RF/XGB 不需要但做了也没坏处)
  # ============================================================
  if (!is.null(config$scale_features) && config$scale_features == TRUE) {
    cat("[Data] 执行 Z-Score 标准化 (Train Mean/SD -> Apply to Test)...\n")
    
    # 计算训练集的均值和标准差
    train_means <- apply(X_train, 2, mean)
    train_sds   <- apply(X_train, 2, sd)
    
    # 极小值保护 (防止除以0)
    train_sds[train_sds == 0] <- 1 
    
    # 应用到训练集
    X_train_scaled <- scale(X_train, center = train_means, scale = train_sds)
    
    # 应用到测试集 (关键：用训练集的参数！)
    X_test_scaled  <- scale(X_test, center = train_means, scale = train_sds)
    
    # 转回 Dataframe
    X_train <- as.data.frame(X_train_scaled)
    X_test  <- as.data.frame(X_test_scaled)
  }
  
  return(list(
    X_train = X_train,
    y_train = train_set$y_target,
    X_test  = X_test,
    y_test  = test_set$y_target,
    raw_train_set = train_set,
    raw_test_set  = test_set,
    feature_cols  = final_features,
    special_cols  = list(momentum = "M4", value = "P4")
  ))
}