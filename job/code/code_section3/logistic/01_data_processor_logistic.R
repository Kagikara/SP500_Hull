# ============================================================
# 模块 1: Logistic 数据处理
# 文件路径: logistic/01_data_processor_logistic.R
# ============================================================

library(dplyr)
library(tidyr)
library(zoo)

process_data_logistic <- function(file_path, config) {
  
  cat("[Data-Logit] 读取数据:", file_path, "...\n")
  if (!file.exists(file_path)) stop("找不到文件")
  raw_data <- read.csv(file_path, stringsAsFactors = FALSE)
  
  if("date_id" %in% names(raw_data)) {
    data_period <- raw_data %>% filter(date_id >= 1006)
  } else {
    data_period <- raw_data[-(1:1006), ]
  }
  
  p_col_names <- names(raw_data)[config$p_cols]
  v_col_names <- names(raw_data)[config$v_cols]
  target_col  <- "market_forward_excess_returns"
  
  # 构造基础数据集 (包含 date_id, target, 和所有 P/V 原始列)
  # 注意：这里我们保留原始列，为了后面做 Imputation
  df_raw <- data_period %>% 
    select(date_id, all_of(target_col), all_of(p_col_names), all_of(v_col_names))
  
  # 构造 Target (0/1)
  df_raw$y <- ifelse(df_raw[[target_col]] >= 0, 1, 0)
  
  # 先切分数据集，再做特征工程
  # ============================================================
  cat("[Data-Logit] 执行时序切分 (Train End:", config$train_end_id, ")...\n")
  
  train_raw <- df_raw %>% filter(date_id <= config$train_end_id)
  test_raw  <- df_raw %>% filter(date_id > config$train_end_id & date_id <= config$test_end_id)
  
  if(nrow(train_raw) == 0) stop("训练集为空")
  
  #  缺失值填补
  # ============================================================
  cat("[Data-Logit] 执行缺失值填补...\n")
  
  impute_cols <- c(p_col_names, v_col_names)
  impute_vals <- sapply(train_raw[impute_cols], median, na.rm = TRUE)
  
  apply_impute <- function(df, vals) {
    for(col in names(vals)) {
      df[[col]][is.na(df[[col]])] <- vals[col]
    }
    return(df)
  }
  
  train_imputed <- apply_impute(train_raw, impute_vals)
  test_imputed  <- apply_impute(test_raw, impute_vals)
  
  # 特征合成
  # ============================================================
  cat("[Data-Logit] 计算合成因子 P_star & V_star...\n")
  
  calc_factors <- function(df) {
    df$P_star <- rowMeans(df[p_col_names])
    df$V_star <- rowMeans(df[v_col_names])
    return(df %>% select(date_id, y, P_star, V_star))
  }
  
  train_factors <- calc_factors(train_imputed)
  test_factors  <- calc_factors(test_imputed)
  
  # 标准化 (Z-Score)
  # ============================================================
  cat("[Data-Logit] 执行标准化\n")
  
  stats <- list(
    p_mean = mean(train_factors$P_star), p_sd = sd(train_factors$P_star),
    v_mean = mean(train_factors$V_star), v_sd = sd(train_factors$V_star)
  )
  
  standardize <- function(x, mu, sigma) { (x - mu) / sigma }
  
  train_factors$P_std <- standardize(train_factors$P_star, stats$p_mean, stats$p_sd)
  train_factors$V_std <- standardize(train_factors$V_star, stats$v_mean, stats$v_sd)
  
  test_factors$P_std  <- standardize(test_factors$P_star, stats$p_mean, stats$p_sd)
  test_factors$V_std  <- standardize(test_factors$V_star, stats$v_mean, stats$v_sd)
  
  # 清理最终包含 NA 的行 (如标准化产生 NaN 或 Target 缺失)
  train_final <- na.omit(train_factors)
  test_final  <- na.omit(test_factors)
  
  cat("[Data-Logit] 处理完成. Train:", nrow(train_final), "| Test:", nrow(test_final), "\n")
  
  return(list(
    train = train_final,
    test = test_final,
    raw_test_set = test_raw, # 保留原始测试集以便后续获取收益率
    stats = stats
  ))
}