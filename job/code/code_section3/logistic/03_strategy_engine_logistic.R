# ============================================================
# 模块 3: Logistic 策略引擎 (适配 Logit 输入 + Platt)
# 文件路径: logistic/03_strategy_engine_logistic.R
# ============================================================

library(zoo)
library(dplyr)
library(ggplot2)

run_logistic_backtest <- function(data_bundle, model_results, config) {
  
  cat("[Strategy-Logit] 启动策略回测...\n")
  
  p_target_vol     <- if(is.null(config$target_vol)) 0.15 else config$target_vol
  p_max_leverage   <- if(is.null(config$max_leverage)) 2.0 else config$max_leverage
  p_trans_cost     <- if(is.null(config$transaction_cost)) 0.001 else config$transaction_cost
  p_roll_window    <- 60
  p_prob_slope     <- 5 
  
  # 1. 概率校准
  # ============================================================
  # 这里的 raw_test_set 是我们在 processor 里特意保留的原始数据
  test_set <- data_bundle$raw_test_set
  
  # 确保行数对齐 (因为 processor 里可能有 na.omit)
  # 我们需要找到 model_results 对应的 date_id
  processed_test <- data_bundle$test
  
  # 只保留处理后还存在的行
  test_set <- test_set %>% filter(date_id %in% processed_test$date_id)
  
  test_scores <- as.numeric(model_results$test_scores)
  
  if(nrow(test_set) != length(test_scores)) {
    stop("错误: 测试集行数与预测结果长度不一致，请检查 na.omit 逻辑")
  }
  
  # 应用校准
  cat("[Strategy-Logit] 应用 Platt Scaling...\n")
  platt_params <- model_results$platt_params
  test_probs <- algo_platt_predict(test_scores, platt_params)
  test_set$pred_prob <- test_probs
  
  # 2. 获取市场收益率 (用于计算)
  # ============================================================
  # processor 保留了原始列，直接用
  mkt_col <- "market_forward_excess_returns"
  if(!mkt_col %in% names(test_set)) stop("找不到市场收益率列")
  
  test_set$market_ret <- test_set[[mkt_col]]
  
  # 3. 仓位与风控 (复用标准逻辑)
  # ============================================================
  cat("[Strategy-Logit] 计算仓位...\n")
  
  # 自适应中枢
  rolling_median <- rollapply(test_set$pred_prob, width = p_roll_window, 
                              FUN = median, fill = NA, align = "right")
  test_set$adaptive_center <- dplyr::lag(rolling_median, n = 1)
  test_set$adaptive_center[is.na(test_set$adaptive_center)] <- 0.5
  
  # 信号
  signal_raw <- (test_set$pred_prob - test_set$adaptive_center) * p_prob_slope
  base_pos <- pmax(0, pmin(1.5, signal_raw / 0.1)) # Long Only 示例
  
  # 波动率控制
  vol_zoo <- rollapply(test_set$market_ret, width = 20, FUN = sd, fill = NA, align = "right") * sqrt(252)
  current_vol <- as.numeric(vol_zoo)
  current_vol[is.na(current_vol) | current_vol < 0.01] <- p_target_vol
  vol_scaler <- p_target_vol / current_vol
  
  # 最终仓位
  test_set$position <- pmin(base_pos * vol_scaler, p_max_leverage)
  test_set$position[is.na(test_set$position)] <- 0
  
  # 4. 绩效核算 (含成本)
  # ============================================================
  pos_change <- diff(c(0, test_set$position))
  transaction_costs <- abs(pos_change) * p_trans_cost
  
  test_set$gross_ret <- test_set$position * test_set$market_ret
  test_set$net_ret   <- test_set$gross_ret - transaction_costs
  
  cat(sprintf("  - [Cost] 总成本损耗: %.2f%%\n", sum(transaction_costs)*100))
  
  # 统计与保存
  calc_metrics <- function(ret) {
    ret <- as.numeric(ret)
    total <- prod(1 + ret, na.rm=TRUE) - 1
    ann <- (1 + total)^(252/length(ret)) - 1
    vol <- sd(ret, na.rm=TRUE) * sqrt(252)
    sharpe <- if(vol==0) 0 else ann/vol
    dd <- min((cumprod(1+ret)/cummax(cumprod(1+ret)))-1, na.rm=TRUE)
    return(c(Ann_Return=ann, Sharpe=sharpe, Max_DD=dd))
  }
  
  res_strat <- calc_metrics(test_set$net_ret)
  res_mkt   <- calc_metrics(test_set$market_ret)
  
  metrics_df <- data.frame(Metric=names(res_strat), Market=round(res_mkt,4), Strategy=round(res_strat,4))
  
  cat("\n----------- Logistic 修正版绩效 (费后) -----------\n")
  print(metrics_df)
  
  write.csv(metrics_df, file.path(config$output_dir, "logistic_metrics_fixed.csv"), row.names = FALSE)
  
  # 绘图
  plot_df <- data.frame(
    Time = 1:nrow(test_set),
    Market = cumprod(1 + test_set$market_ret),
    Strategy = cumprod(1 + test_set$net_ret)
  )
  if("date_id" %in% names(test_set)) plot_df$Time <- test_set$date_id
  
  p <- ggplot(plot_df, aes(x = Time)) +
    geom_line(aes(y = Market, color = "Market"), alpha = 0.6) +
    geom_line(aes(y = Strategy, color = "Logistic (Net)"), linewidth = 1) +
    scale_color_manual(values = c("Market"="grey", "Logistic (Net)"="purple")) +
    labs(title = "Logistic Strategy (Imputed & Calibrated)",
         subtitle = paste("Sharpe:", round(res_strat['Sharpe'], 2)),
         y = "Net Value") +
    theme_minimal()
  
  ggsave(file.path(config$output_dir, "logistic_equity_fixed.png"), p, width=8, height=5)
  
  return(list(metrics = metrics_df, results = test_set))
}