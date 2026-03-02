# ============================================================
# 模块 3: RF回测引擎
# ============================================================

library(zoo)
library(dplyr)
library(ggplot2)

run_strategy_backtest_rf <- function(data_bundle, model_bundle, config) {
  
  cat("[Strategy-RF] 启动策略回测...\n")
  p_target_vol   <- if(is.null(config$target_vol)) 0.15 else config$target_vol
  p_max_leverage <- if(is.null(config$max_leverage)) 2.0 else config$max_leverage
  p_trend_window <- if(is.null(config$trend_window)) 60 else config$trend_window
  p_roll_window  <- if(is.null(config$prob_roll_window)) 60 else config$prob_roll_window
  p_prob_slope   <- if(is.null(config$prob_slope)) 10 else config$prob_slope
  p_trans_cost   <- if(is.null(config$transaction_cost)) 0.001 else config$transaction_cost
  
  test_set <- data_bundle$raw_test_set
  test_scores <- as.numeric(model_bundle$test_scores)
  
  if (is.null(model_bundle$platt_params)) stop("错误: 缺少 Platt 参数")
  
  platt_params <- model_bundle$platt_params
  cat(sprintf("  - [Calibration] Platt参数: A=%.4f, B=%.4f\n", 
              platt_params$A, platt_params$B))
  
  test_probs <- algo_platt_predict(test_scores, platt_params)
  test_probs[is.na(test_probs) | is.infinite(test_probs)] <- 0.5
  test_set$pred_prob <- test_probs
  
  # ============================================================
  # 2. 自适应中枢
  # ============================================================
  cat("[Strategy-RF] 计算自适应中枢\n")
  n <- nrow(test_set)
  rolling_median_raw <- rollapply(
    test_set$pred_prob, 
    width = p_roll_window, 
    FUN = median, 
    fill = NA, 
    align = "right"
  )
  test_set$adaptive_center <- dplyr::lag(rolling_median_raw, n = 1)
  warmup_indices <- which(is.na(test_set$adaptive_center))
  
  if(length(warmup_indices) > 0) {
    warmup_value <- median(test_set$pred_prob[1:min(p_roll_window, n)], na.rm=TRUE)
    test_set$adaptive_center[warmup_indices] <- warmup_value
    cat(sprintf("  - 预热期 (%d天) 使用固定中枢: %.4f\n", 
                length(warmup_indices), warmup_value))
  }
  
  if(any(is.na(test_set$adaptive_center))) {
    warning("自适应中枢仍有NA，全部填充为0.5")
    test_set$adaptive_center[is.na(test_set$adaptive_center)] <- 0.5
  }
  
  # ============================================================
  # 3. 风控因子
  # ============================================================
  cat("[Strategy-RF] 计算风控因子...\n")
  
  past_returns <- dplyr::lag(test_set$market_forward_excess_returns, 1)
  past_returns[is.na(past_returns)] <- 0
  
  vol_raw <- rollapply(past_returns, width = 20, FUN = sd, fill = NA, align = "right") * sqrt(252)
  current_vol <- dplyr::lag(vol_raw, 1)
  current_vol[is.na(current_vol) | current_vol <= 0.01] <- p_target_vol

  price_index <- cumprod(1 + past_returns)
  ma_trend_raw <- rollapply(price_index, width = p_trend_window, FUN = mean, fill = NA, align = "right")
  ma_trend <- dplyr::lag(ma_trend_raw, 1)
  ma_trend[is.na(ma_trend)] <- price_index[is.na(ma_trend)]
  
  is_bull <- ifelse(price_index > ma_trend, 1, 0)
  
  # ============================================================
  # 4. 仓位计算
  # ============================================================
  cat("[Strategy-RF] 计算动态仓位...\n")
  
  # Alpha引擎: 基于概率与中枢的偏离
  signal_score <- 0.5 + (test_set$pred_prob - test_set$adaptive_center) * p_prob_slope
  base_pos <- pmax(0.2, pmin(1.5, signal_score))
  
  # Risk引擎: 波动率倒数缩放
  vol_scaler <- p_target_vol / current_vol
  
  # Trend引擎: 趋势倍率
  trend_mult <- ifelse(is_bull == 1, 1.0, 0.6)
  
  # 最终仓位
  raw_pos <- base_pos * vol_scaler * trend_mult
  test_set$position <- pmin(raw_pos, p_max_leverage)
  test_set$position[is.na(test_set$position)] <- 0
  
  cat("  - 平均持仓:", round(mean(test_set$position), 2), "\n")
  
  # ============================================================
  # 5. 绩效统计 (含交易成本)
  # ============================================================
  pos_change <- diff(c(0, test_set$position))
  turnover_cost <- abs(pos_change) * p_trans_cost
  
  test_set$gross_return <- test_set$position * test_set$market_forward_excess_returns
  test_set$strategy_return <- test_set$gross_return - turnover_cost
  
  cat(sprintf("  - [Cost] 总交易成本: %.2f%%\n", sum(turnover_cost)*100))
  
  calc_metrics <- function(ret) {
    ret <- as.numeric(ret)
    total <- prod(1 + ret, na.rm=TRUE) - 1
    ann <- (1 + total)^(252/length(ret)) - 1
    vol <- sd(ret, na.rm=TRUE) * sqrt(252)
    sharpe <- if(vol==0) 0 else ann/vol
    cum <- cumprod(1 + ret)
    dd <- min((cum/cummax(cum))-1, na.rm=TRUE)
    return(c(Ann_Return=ann, Sharpe=sharpe, Max_DD=dd))
  }
  
  res_strat <- calc_metrics(test_set$strategy_return)
  res_mkt   <- calc_metrics(test_set$market_forward_excess_returns)
  
  metrics_df <- data.frame(
    Metric=names(res_strat), 
    Market=round(res_mkt,4), 
    Strategy=round(res_strat,4)
  )
  
  cat("----------- RF 回测绩效报告 (费后) -----------\n")
  print(metrics_df)
  
  write.csv(metrics_df, file.path(config$output_dir, "rf_performance_metrics.csv"), row.names = FALSE)
  
  plot_df <- data.frame(
    Time = if("date_id" %in% names(test_set)) test_set$date_id else 1:n,
    Market = cumprod(1 + test_set$market_forward_excess_returns),
    Strategy = cumprod(1 + test_set$strategy_return)
  )
  
  p <- ggplot(plot_df, aes(x = Time)) +
    geom_line(aes(y = Market, color = "Market"), alpha = 0.6) +
    geom_line(aes(y = Strategy, color = "RF Strategy (Net)"), linewidth = 1) +
    scale_color_manual(values = c("Market"="grey", "RF Strategy (Net)"="blue")) +
    labs(title = "Random Forest Strategy (Strict Anti-Leakage)",
         subtitle = paste("Sharpe:", round(res_strat['Sharpe'], 2)),
         y = "Cumulative Return") +
    theme_minimal()
  
  ggsave(file.path(config$output_dir, "rf_equity_curve_final.png"), p, width=10, height=6)
  
  return(list(
    results_df = test_set, 
    metrics = metrics_df, 
    plot = p,
    platt_params = platt_params
  ))
}