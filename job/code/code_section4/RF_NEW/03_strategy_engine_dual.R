## ============================================================
# 模块 3: 双流策略引擎 (XGB Trend + RF Sniper) [修复版]
# 文件路径: code_section4/RF_NEW/03_strategy_engine_dual.R
# ============================================================

library(zoo)
library(dplyr)
library(ggplot2)

# 辅助函数: 计算滚动排名
calc_rolling_rank <- function(x, width) {
  rng_rank <- rollapply(x, width, function(z) {
    current <- z[length(z)]
    mean(z <= current, na.rm = TRUE)
  }, fill = NA, align = "right")
  
  if (any(is.na(rng_rank))) {
    valid_idx <- which(!is.na(rng_rank))[1]
    if(!is.na(valid_idx) && valid_idx > 1) {
      rng_rank[1:(valid_idx-1)] <- 0.5 
    }
  }
  return(rng_rank)
}

run_strategy_dual <- function(data_bundle, bundle_trend, bundle_sniper, config) {
  
  cat("[Strategy-Dual] 启动双模策略 (Trend=XGB, Sniper=RF)...\n")
  
  # --- 0. 参数初始化 (修复报错的关键) ---
  sniper <- config$sniper_mode
  
  # 从 Config 提取通用参数，若无则给默认值
  p_rank_window  <- if(is.null(config$rank_window)) 250 else config$rank_window
  p_trans_cost   <- if(is.null(config$transaction_cost)) 0.001 else config$transaction_cost
  p_target_vol   <- if(is.null(config$target_vol)) 0.15 else config$target_vol
  p_max_leverage <- if(is.null(config$max_leverage)) 2.0 else config$max_leverage
  
  cat(sprintf("  - 参数检查: RankWin=%d, Cost=%.4f, Vol=%.2f\n", 
              p_rank_window, p_trans_cost, p_target_vol))

  # --- 1. 准备概率数据 (双流) ---
  test_set <- data_bundle$raw_test_set
  
  # A. 提取 XGB 概率 (用于牛市 Trend)
  cat("  - 校准 XGB 概率 (Trend)...\n")
  xgb_scores <- as.numeric(bundle_trend$test_scores)
  xgb_platt  <- bundle_trend$platt_params
  test_set$prob_trend <- algo_platt_predict(xgb_scores, xgb_platt)
  
  # B. 提取 RF 概率 (用于熊市 Sniper)
  cat("  - 校准 RF 概率 (Sniper)...\n")
  rf_scores <- as.numeric(bundle_sniper$test_scores)
  rf_platt  <- bundle_sniper$platt_params
  test_set$prob_sniper <- algo_platt_predict(rf_scores, rf_platt)
  
  # --- 2. 市场状态感知 (Regime Detection) ---
  # 使用 MA60 判断牛熊
  raw_train_mkt <- data_bundle$raw_train_set$market_forward_excess_returns
  raw_test_mkt  <- data_bundle$raw_test_set$market_forward_excess_returns
  full_mkt_index <- cumprod(1 + c(raw_train_mkt, raw_test_mkt))
  
  # 计算均线
  trend_win <- if(is.null(sniper$trend_window)) 60 else sniper$trend_window
  full_ma <- rollmean(full_mkt_index, k = trend_win, fill = NA, align = "right")
  
  is_bull_raw <- (full_mkt_index > full_ma)
  is_bull_lagged <- dplyr::lag(is_bull_raw, 1) # 防偷看
  
  n_train <- length(raw_train_mkt)
  n_test  <- length(raw_test_mkt)
  
  # 切分回测试集
  test_set$is_bull <- is_bull_lagged[(n_train + 1) : (n_train + n_test)]
  test_set$is_bull[is.na(test_set$is_bull)] <- FALSE
  
  # --- 3. 构建 Sniper 动态阈值 (仅基于 RF 的历史) ---
  cat("  - 计算 RF Sniper 动态阈值...\n")
  if(is.null(bundle_sniper$train_cv_probs)) stop("RF Bundle 缺少 CV Probs")
  
  rf_full_probs <- c(bundle_sniper$train_cv_probs, test_set$prob_sniper)
  
  rf_thresh_raw <- rollapply(
    rf_full_probs, 
    width = sniper$lookback_window, 
    FUN = function(x) quantile(x, probs = sniper$open_quantile, na.rm = TRUE),
    fill = NA, align = "right"
  )
  rf_thresh_lagged <- dplyr::lag(rf_thresh_raw, 1)
  test_sniper_thresh <- rf_thresh_lagged[(n_train + 1) : (n_train + n_test)]
  
  # 最终 Sniper 阈值 (带硬保底)
  test_set$thresh_sniper <- pmax(test_sniper_thresh, sniper$hard_floor_open, na.rm=TRUE)
  
  # --- 4. 双模状态机 (Hybrid State Machine) ---
  cat("[Strategy-Dual] 执行混合信号路由...\n")
  
  rf_position_signal <- numeric(n_test)
  current_state <- 0 # 0=Base, 1=Boost, -1=Bailout
  
  # 提取向量加速
  vec_prob_trend  <- test_set$prob_trend   # XGB
  vec_prob_sniper <- test_set$prob_sniper  # RF
  vec_thresh_snp  <- test_set$thresh_sniper
  vec_is_bull     <- test_set$is_bull
  
  trend_floor <- if(is.null(sniper$trend_threshold)) 0.50 else sniper$trend_threshold
  
  for(i in 1:n_test) {
    is_bull <- vec_is_bull[i]
    
    if (is_bull) {
      # === 模式 A: 牛市 (Trend - XGB) ===
      prob_t <- vec_prob_trend[i]
      
      if (current_state == 0) {
        if (prob_t > trend_floor) current_state <- 1
        if (prob_t < 0.45) current_state <- -1
      } else if (current_state == 1) {
        if (prob_t < 0.48) current_state <- 0
      } else if (current_state == -1) {
        if (prob_t > 0.50) current_state <- 0
      }
      
    } else {
      # === 模式 B: 熊市 (Sniper - RF) ===
      prob_s <- vec_prob_sniper[i]
      thr_s  <- vec_thresh_snp[i]
      
      if (current_state == 0) {
        if (!is.na(thr_s) && prob_s > thr_s) {
          current_state <- 1
        } else if (prob_s < 0.45) {
          current_state <- -1
        }
      } else if (current_state == 1) {
        if (prob_s < 0.52) current_state <- 0
      } else if (current_state == -1) {
        if (prob_s > 0.52) current_state <- 0
      }
    }
    
    # 仓位映射
    pos_base  <- sniper$pos_base
    pos_boost <- sniper$pos_boost
    
    if (current_state == 1) {
      rf_position_signal[i] <- pos_boost
    } else if (current_state == -1) {
      rf_position_signal[i] <- 0.0
    } else {
      rf_position_signal[i] <- if(is_bull) pos_base else (pos_base * 0.8)
    }
  }
  
  test_set$signal_final <- rf_position_signal
  
  # --- 5. 因子互作滤网 (M-P Filter) ---
  cat("[Strategy-Dual] 计算 M-P 互作滤网...\n")
  col_m <- data_bundle$special_cols$momentum
  col_p <- data_bundle$special_cols$value
  
  # [修复点]: 现在 p_rank_window 已经定义了
  test_set$rank_m <- calc_rolling_rank(test_set[[col_m]], p_rank_window)
  test_set$rank_p <- calc_rolling_rank(test_set[[col_p]], p_rank_window)
  
  test_set$filter_mp <- 1.0
  
  # 规则1: 崩盘保护 (M < 20%)
  mask_crash <- test_set$rank_m < 0.20
  test_set$filter_mp[mask_crash] <- 0.0
  
  # 规则2: 震荡观察
  mask_chop <- (test_set$rank_m >= 0.20) & (test_set$rank_m < 0.70)
  test_set$filter_mp[mask_chop] <- 0.5
  
  # 规则3: 估值泡沫
  mask_bubble <- test_set$rank_p > 0.90
  test_set$filter_mp[mask_bubble & !mask_crash] <- 0.5
  
  # 规则4: 戴维斯双击
  mask_boost <- (test_set$rank_m >= 0.66) & (test_set$rank_p < 0.70)
  test_set$filter_mp[mask_boost] <- 2.0
  
  # --- 6. 风控与合成 ---
  vol_zoo <- rollapply(test_set$market_forward_excess_returns, 
                       width = 20, FUN = sd, fill = NA, align = "right") * sqrt(252)
  current_vol <- as.numeric(vol_zoo)
  current_vol[is.na(current_vol) | current_vol <= 0] <- p_target_vol
  test_set$regime_vol <- p_target_vol / current_vol
  
  # 合成 (使用 signal_final)
  raw_pos <- test_set$signal_final * test_set$filter_mp * test_set$regime_vol
  test_set$position <- pmin(raw_pos, p_max_leverage)
  test_set$position[is.na(test_set$position)] <- 0
  
  # --- 7. 绩效统计 (含 Cost) ---
  pos_change <- diff(c(0, test_set$position))
  turnover_cost <- abs(pos_change) * p_trans_cost
  
  test_set$gross_return <- test_set$position * test_set$market_forward_excess_returns
  test_set$strategy_return <- test_set$gross_return - turnover_cost
  
  cat(sprintf("  - [Cost] 交易成本: %.2f%%\n", sum(turnover_cost)*100))
  
  # 绩效函数
  calc_metrics <- function(ret) {
    ret <- as.numeric(ret)
    total <- prod(1 + ret, na.rm=TRUE) - 1
    ann <- (1 + total)^(252/length(ret)) - 1
    vol <- sd(ret, na.rm=TRUE) * sqrt(252)
    sharpe <- if(vol==0) 0 else ann/vol
    dd <- min((cumprod(1+ret)/cummax(cumprod(1+ret)))-1, na.rm=TRUE)
    return(c(Ann_Return=ann, Sharpe=sharpe, Max_DD=dd))
  }
  
  res_strat <- calc_metrics(test_set$strategy_return)
  res_mkt   <- calc_metrics(test_set$market_forward_excess_returns)
  
  metrics_df <- data.frame(Metric=names(res_strat), Market=round(res_mkt,4), Strategy=round(res_strat,4))
  cat("\n----------- Hybrid MoE 策略绩效 -----------\n")
  print(metrics_df)
  
  write.csv(metrics_df, file.path(config$output_dir, "hybrid_metrics.csv"), row.names = FALSE)
  
  # --- 8. 绘图 (双模版) ---
  plot_df <- data.frame(
    Time = 1:nrow(test_set),
    Market = cumprod(1 + test_set$market_forward_excess_returns),
    Strategy = cumprod(1 + test_set$strategy_return),
    State = factor(test_set$is_bull, labels=c("Bear/Chop", "Bull")),
    Prob_RF = test_set$prob_sniper,
    Prob_XGB = test_set$prob_trend
  )
  if("date_id" %in% names(test_set)) plot_df$Time <- test_set$date_id
  
  p1 <- ggplot(plot_df, aes(x = Time)) +
    geom_line(aes(y = Market, color = "Market"), alpha = 0.5) +
    geom_line(aes(y = Strategy, color = "Hybrid (Net)"), linewidth = 1) +
    scale_color_manual(values = c("Market"="grey", "Hybrid (Net)"="purple")) +
    labs(title = "Hybrid MoE Strategy: XGB(Bull) + RF(Bear)",
         subtitle = paste("Sharpe:", round(res_strat['Sharpe'], 2)),
         y = "Net Value") +
    theme_minimal()
  
  ggsave(file.path(config$output_dir, "hybrid_equity.png"), p1, width=8, height=5)
  
  return(list(
    results_df = test_set, 
    metrics = metrics_df, 
    plot = p1
  ))
}