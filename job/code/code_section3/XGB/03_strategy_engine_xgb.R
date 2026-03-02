# ============================================================
# 模块 3: XGB 专用策略回测引擎 (已修复 A4/A5 级缺陷)
# 文件路径: XGB/03_strategy_engine_xgb.R
# ============================================================

library(zoo)
library(dplyr)
library(ggplot2)

run_strategy_backtest_xgb <- function(data_bundle, model_bundle, config) {
  
  cat("[Strategy-XGB] 启动策略回测...\n")
  
  # 0. 参数提取
  p_target_vol   <- if(is.null(config$target_vol)) 0.15 else config$target_vol
  p_max_leverage <- if(is.null(config$max_leverage)) 2.0 else config$max_leverage
  p_trend_window <- if(is.null(config$trend_window)) 60 else config$trend_window
  p_roll_window  <- if(is.null(config$prob_roll_window)) 60 else config$prob_roll_window
  p_prob_slope   <- if(is.null(config$prob_slope)) 10 else config$prob_slope
  # [新增] 交易成本
  p_trans_cost   <- if(is.null(config$transaction_cost)) 0.001 else config$transaction_cost
  
  cat("  - 策略参数: Vol=", p_target_vol, " Lev=", p_max_leverage, " Cost=", p_trans_cost, "\n")
  
  # 1. 概率校准 (应用训练集参数)
  # ------------------------------------------------------------
  test_set <- data_bundle$raw_test_set
  test_scores <- as.numeric(model_bundle$test_scores)
  
  if (length(test_scores) == 0) stop("错误: 未接收到 XGB 预测得分！")
  
  # [修复 A4]: 严禁在测试集上 fit
  if (is.null(model_bundle$platt_params)) stop("错误: 缺少 Platt 校准参数")
  
  platt_params <- model_bundle$platt_params
  cat(sprintf("[Strategy-XGB] 应用校准参数 A=%.4f, B=%.4f\n", platt_params$A, platt_params$B))
  
  test_probs <- algo_platt_predict(test_scores, platt_params)
  test_set$pred_prob <- test_probs
  
  cat("  - 校准后概率范围:", round(min(test_probs), 4), "~", round(max(test_probs), 4), "\n")
  
  # 2. 计算风控因子
  # ------------------------------------------------------------
  cat("[Strategy-XGB] 计算风控因子...\n")
  
  vol_zoo <- rollapply(test_set$market_forward_excess_returns, 
                       width = 20, FUN = sd, fill = NA, align = "right") * sqrt(252)
  current_vol <- as.numeric(vol_zoo)
  current_vol[is.na(current_vol) | current_vol <= 0] <- p_target_vol
  
  price_index <- cumprod(1 + test_set$market_forward_excess_returns)
  ma_trend <- rollapply(price_index, width = p_trend_window, FUN = mean, fill = NA, align = "right")
  ma_trend[is.na(ma_trend)] <- price_index[is.na(ma_trend)]
  is_bull <- ifelse(price_index > ma_trend, 1, 0)
  
  # 3. 仓位计算
  # ------------------------------------------------------------
  cat("[Strategy-XGB] 计算动态仓位...\n")
  
  rolling_median <- rollapply(test_set$pred_prob, width = p_roll_window, 
                              FUN = median, fill = NA, align = "right")
  test_set$adaptive_center <- dplyr::lag(rolling_median, n = 1)
  
  default_center <- median(test_set$pred_prob, na.rm = TRUE)
  test_set$adaptive_center[is.na(test_set$adaptive_center)] <- default_center
  
  signal_score <- 0.5 + (test_set$pred_prob - test_set$adaptive_center) * p_prob_slope
  base_pos <- pmax(0.2, pmin(1.5, signal_score))
  
  vol_scaler <- p_target_vol / current_vol
  trend_mult <- ifelse(is_bull == 1, 1.0, 0.6)
  
  raw_pos <- base_pos * vol_scaler * trend_mult
  test_set$position <- pmin(raw_pos, p_max_leverage)
  test_set$position[is.na(test_set$position)] <- 0
  
  cat("  - 平均持仓:", round(mean(test_set$position), 2), "\n")
  
  # 4. 绩效统计 (含成本)
  # ------------------------------------------------------------
  # [修复 A5]: 扣除交易成本
  pos_change <- diff(c(0, test_set$position))
  turnover_cost <- abs(pos_change) * p_trans_cost
  
  test_set$gross_return <- test_set$position * test_set$market_forward_excess_returns
  test_set$strategy_return <- test_set$gross_return - turnover_cost
  
  cat(sprintf("  - [Cost] 总交易成本损耗: %.2f%%\n", sum(turnover_cost)*100))
  
  # 指标计算
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
  
  metrics_df <- data.frame(Metric=names(res_strat), Market=round(res_mkt,4), Strategy=round(res_strat,4))
  cat("\n----------- XGB 回测绩效报告 (费后) -----------\n")
  print(metrics_df)
  
  write.csv(metrics_df, file.path(config$output_dir, "xgb_performance_metrics.csv"), row.names = FALSE)
  
  # 绘图
  plot_df <- data.frame(
    Time = 1:nrow(test_set),
    Market = cumprod(1 + test_set$market_forward_excess_returns),
    Strategy = cumprod(1 + test_set$strategy_return)
  )
  if("date_id" %in% names(test_set)) plot_df$Time <- test_set$date_id
  
  p <- ggplot(plot_df, aes(x = Time)) +
    geom_line(aes(y = Market, color = "Market"), alpha = 0.6) +
    geom_line(aes(y = Strategy, color = "XGB Strategy (Net)"), linewidth = 1) +
    scale_color_manual(values = c("Market"="grey", "XGB Strategy (Net)"="blue")) +
    labs(title = "XGBoost Strategy Performance (Net of Cost)",
         subtitle = paste("Sharpe:", round(res_strat['Sharpe'], 2)),
         y = "Net Value") +
    theme_minimal()
  
  ggsave(file.path(config$output_dir, "xgb_equity_curve.png"), p, width=8, height=5)
  
  return(list(
    results_df = test_set, 
    metrics = metrics_df, 
    plot = p,
    platt_params = platt_params
  ))
}