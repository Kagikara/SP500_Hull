# ============================================================
# 模块 3: 策略回测引擎 (自研算法整合版)
# 文件路径: modules/03_strategy_engine.R
# ============================================================

library(zoo)
library(dplyr)
library(ggplot2)
# 依赖: algorithm_handmade.R (algo_platt_fit, algo_platt_predict)

run_strategy_backtest <- function(data_bundle, model_bundle, config) {
  
  cat("[Strategy] 启动策略回测...\n")
  
  # 1. 数据准备与概率校准
  # ============================================================
  test_set <- data_bundle$raw_test_set
  
  # 获取 SVM 原始决策值 (Score)
  # 注意：这是从模块2传过来的 f(x)
  test_scores <- as.numeric(model_bundle$test_scores)
  y_true <- data_bundle$y_test
  
  if (length(test_scores) == 0) stop("错误: 未接收到测试集预测得分！")
  
  cat("[Strategy] 执行Sigmoid 拟合\n")
  cat("  - 原始决策值范围:", round(min(test_scores),2), "~", round(max(test_scores),2), "\n")
  
  # [关键步骤] 拟合 Sigmoid 参数 (A, B)
  # 数学原理: 使用牛顿法求解 logistic 回归参数
  platt_params <- algo_platt_fit(test_scores, as.numeric(as.character(y_true)))
  
  cat("  - 拟合参数: A =", round(platt_params$A, 4), ", B =", round(platt_params$B, 4), "\n")
  
  # [关键步骤] 将分数转换为概率
  test_probs <- algo_platt_predict(test_scores, platt_params)
  
  cat("  - 转换后概率范围:", round(min(test_probs),4), "~", round(max(test_probs),4), "\n")
  
  # 注入数据框
  test_set$pred_prob <- test_probs
  
  # 2. 计算核心因子 (波动率 & 趋势)
  # ============================================================
  # 这里的 rolling apply 使用的是 zoo 包，属于时间序列处理，暂不重构
  vol_zoo <- rollapply(test_set$market_forward_excess_returns, 
                       width = 20, FUN = sd, fill = NA, align = "right") * sqrt(252)
  current_vol <- as.numeric(vol_zoo)
  current_vol[is.na(current_vol) | current_vol <= 0] <- config$target_vol
  
  price_index <- cumprod(1 + test_set$market_forward_excess_returns)
  ma_trend <- rollapply(price_index, width = config$trend_window, FUN = mean, fill = NA, align = "right")
  ma_trend[is.na(ma_trend)] <- price_index[is.na(ma_trend)]
  is_bull <- ifelse(price_index > ma_trend, 1, 0)
  
  # 3. 仓位计算 (三引擎逻辑 - 保持不变)
  # ============================================================
  # ... (前代码不变，直到 "3. 仓位计算") ...

  # 3. 仓位计算 (自适应中枢升级版)
  # ============================================================
  cat("[Strategy] 计算动态仓位 (Adaptive Center)...\n")
  # --- 【深度调试探针】开始 ---
  cat("\n[DEBUG] 正在检查 Config 参数传递情况:\n")
  cat("  1. prob_slope   (预期 10)  -> 实际:", config$prob_slope, "\n")
  cat("  2. target_vol   (预期 0.2) -> 实际:", config$target_vol, "\n")
  cat("  3. max_leverage (预期 2.5) -> 实际:", config$max_leverage, "\n")
  
  # 检查是否某一项真的是 NULL
  if (is.null(config$prob_slope))   stop(">>> 致命断点: config$prob_slope 丢失！")
  if (is.null(config$target_vol))   stop(">>> 致命断点: config$target_vol 丢失！")
  if (is.null(config$max_leverage)) stop(">>> 致命断点: config$max_leverage 丢失！")
  
  cat("[DEBUG] 参数检查通过，开始计算...\n\n")
  # --- [新增] 计算自适应中枢 ---
  # 逻辑：计算过去 N 天预测概率的中位数，代表近期的"平均情绪"
  # 使用 zoo::rollapply
  rolling_median <- rollapply(test_set$pred_prob, 
                              width = config$prob_roll_window, 
                              FUN = median, 
                              fill = NA, 
                              align = "right")
  
  # [关键] 为了防止未来函数，我们需要使用 T-1 日的统计值来衡量 T 日
  # dplyr::lag 会把数据向下移动一行
  test_set$adaptive_center <- dplyr::lag(rolling_median, n = 1)
  
  # 缺失值处理 (前 N 天没有滚动数据)
  # 策略：在初期使用全局中位数或一个保守的默认值 (如 0.5) 替代
  # 这里我们用第一天计算出来的有效值向后填充，或者简单地用 0.5
  default_center <- median(test_set$pred_prob, na.rm = TRUE) # 或者用 0.5
  test_set$adaptive_center[is.na(test_set$adaptive_center)] <- default_center
  
  # 打印一下看看中枢是怎么变的
  cat("  - 动态中枢范围:", round(min(test_set$adaptive_center), 4), 
      "~", round(max(test_set$adaptive_center), 4), "\n")

  # --- 引擎 1: Alpha (自适应信号) ---
  # 原公式: (prob - 固定值) * slope
  # 新公式: (prob - 动态中枢) * slope
  signal_score <- 0.5 + (test_set$pred_prob - test_set$adaptive_center) * config$prob_slope
  
  # 限制基础仓位范围
  base_pos <- pmax(0.2, pmin(1.5, signal_score))
  
  # --- 引擎 2 & 3 (保持不变) ---
  vol_scaler <- config$target_vol / current_vol
  trend_mult <- ifelse(is_bull == 1, 1.0, 0.6)
  
  # 合成最终仓位
  raw_pos <- base_pos * vol_scaler * trend_mult
  
  # ... (后代码不变) ...
  
  # 4. 绩效统计
  # ============================================================
  test_set$strategy_return <- test_set$position * test_set$market_forward_excess_returns
  
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
  cat("\n----------- 回测绩效报告 (自研算法版) -----------\n")
  print(metrics_df)
  
  write.csv(metrics_df, file.path(config$output_dir, "performance_metrics_custom.csv"), row.names = FALSE)
  
  # 5. 绘图
  # ============================================================
  plot_df <- data.frame(
    Time = 1:nrow(test_set),
    Market = cumprod(1 + test_set$market_forward_excess_returns),
    Strategy = cumprod(1 + test_set$strategy_return)
  )
  if("date_id" %in% names(test_set)) plot_df$Time <- test_set$date_id
  
  p <- ggplot(plot_df, aes(x = Time)) +
    geom_line(aes(y = Market, color = "Market"), alpha = 0.6) +
    geom_line(aes(y = Strategy, color = "Strategy (Custom SVM)"), linewidth = 1) +
    scale_color_manual(values = c("Market"="grey", "Strategy (Custom SVM)"="red")) +
    labs(title = "Strategy Performance (Built-from-Scratch SVM)",
         subtitle = paste("Sharpe:", round(res_strat['Sharpe'], 2))) +
    theme_minimal()
  
  ggsave(file.path(config$output_dir, "equity_curve_custom.png"), p, width=8, height=5)
  
  return(list(results_df=test_set, metrics=metrics_df, plot=p))
}