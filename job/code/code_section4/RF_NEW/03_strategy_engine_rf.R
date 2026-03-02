# ============================================================
# 模块 3: 策略引擎
# ============================================================

library(zoo)
library(dplyr)
library(ggplot2)

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

run_strategy_backtest_rf <- function(data_bundle, model_bundle, config) {
  
  cat("[Strategy-RF] 启动自适应狙击回测\n")

  sniper <- config$sniper_mode
  p_target_vol   <- if(is.null(config$target_vol)) 0.15 else config$target_vol
  p_max_leverage <- if(is.null(config$max_leverage)) 2.0 else config$max_leverage
  p_trans_cost   <- if(is.null(config$transaction_cost)) 0.001 else config$transaction_cost
  p_rank_window  <- if(is.null(config$rank_window)) 250 else config$rank_window
  
  if(is.null(sniper) || !sniper$active) {
    stop("错误: Config 中未找到 sniper_mode 参数，请检查 main_rf.R 配置")
  }
  
  cat(sprintf("  - [Sniper Config] Lookback=%d, Open_Q=%.2f, Hard_Floor=%.2f\n", 
              sniper$lookback_window, sniper$open_quantile, sniper$hard_floor_open))
  
  test_set <- data_bundle$raw_test_set
  test_scores <- as.numeric(model_bundle$test_scores)
  
  if (length(test_scores) == 0) stop("错误: 无预测得分")
  if (is.null(model_bundle$platt_params)) stop("错误: 缺少 Platt 参数")
  
  platt_params <- model_bundle$platt_params
  cat(sprintf("  - [Calibration] 应用参数: A=%.4f, B=%.4f\n", platt_params$A, platt_params$B))
  
  test_set$pred_prob <- algo_platt_predict(test_scores, platt_params)
  
  cat("[Strategy-RF] 计算动态阈值\n")
  
  if(is.null(model_bundle$train_cv_probs)) {
    stop("错误: model_bundle 中缺少 train_cv_probs。请确保已更新 02_model_trainer_rf_handmade.R")
  }
  train_probs <- model_bundle$train_cv_probs
  test_probs  <- test_set$pred_prob

  full_history_probs <- c(train_probs, test_probs)
  n_train <- length(train_probs)
  n_test  <- length(test_probs)
  
  rolling_threshold_raw <- rollapply(
    full_history_probs, 
    width = sniper$lookback_window, 
    FUN = function(x) quantile(x, probs = sniper$open_quantile, na.rm = TRUE),
    fill = NA, 
    align = "right"
  )
  
  rolling_threshold_lagged <- dplyr::lag(rolling_threshold_raw, 1)

  test_dynamic_thresholds <- rolling_threshold_lagged[(n_train + 1) : (n_train + n_test)]
  
  test_dynamic_thresholds[is.na(test_dynamic_thresholds)] <- sniper$hard_floor_open

  test_set$thresh_open <- pmax(test_dynamic_thresholds, sniper$hard_floor_open)
  
  cat("[Strategy-RF] 执行信号状态机 (模式: 指数增强)...\n")

  pos_base    <- if(is.null(sniper$pos_base)) 0.8 else sniper$pos_base
  pos_boost   <- if(is.null(sniper$pos_boost)) 1.3 else sniper$pos_boost
  pos_bailout <- if(is.null(sniper$pos_bailout)) 0.0 else sniper$pos_bailout
  bail_thr    <- if(is.null(sniper$bailout_thresh)) 0.45 else sniper$bailout_thresh
  
  cat(sprintf("  - [仓位管理] Base=%.1f, Boost=%.1f, Bailout=%.1f (Thr<%.2f)\n", 
              pos_base, pos_boost, pos_bailout, bail_thr))

  rf_position_signal <- numeric(n_test)

  current_state <- 0 
  
  vec_probs <- test_set$pred_prob
  vec_thresh_open <- test_set$thresh_open
  
  for(i in 1:n_test) {
    prob_t   <- vec_probs[i]
    dyn_open <- vec_thresh_open[i]
    
    # ----------------------------------------------------
    # 状态转换逻辑
    # ----------------------------------------------------
    
    if (current_state == 0) {
      # [当前: 底仓] -> 检查是否变强或变弱
      if (prob_t > dyn_open) {
        current_state <- 1 # 触发狙击，加仓
      } else if (prob_t < bail_thr) {
        current_state <- -1 # 信号极差，砍底仓
      }
      
    } else if (current_state == 1) {
      # 检查是否该止盈(退回底仓)
      # 如果概率掉回 0.50 以下，说明多头趋势不再，退守底仓
      if (prob_t < sniper$hard_floor_close) {
        # 二次判断: 如果跌得很惨，直接去逃生；否则只是退回底仓
        if (prob_t < bail_thr) {
          current_state <- -1
        } else {
          current_state <- 0
        }
      }
      # 否则保持增强状态
      
    } else if (current_state == -1) {
      # 检查是否安全了(回补底仓)
      # 只要概率回到 0.50 以上，说明市场企稳，建立底仓
      if (prob_t > sniper$hard_floor_close) {
        # 如果甚至超过了狙击线，直接进增强
        if (prob_t > dyn_open) {
          current_state <- 1
        } else {
          current_state <- 0
        }
      }
    }
  
    if (current_state == 1) {
      rf_position_signal[i] <- pos_boost
    } else if (current_state == -1) {
      rf_position_signal[i] <- pos_bailout
    } else {
      rf_position_signal[i] <- pos_base
    }
  }
  
  test_set$signal_rf <- rf_position_signal

  # --- 因子互作滤网 ---
  cat("[Strategy-RF] 计算 M-P 互作滤网...\n")
  col_m <- data_bundle$special_cols$momentum
  col_p <- data_bundle$special_cols$value
  
  test_set$rank_m <- calc_rolling_rank(test_set[[col_m]], p_rank_window)
  test_set$rank_p <- calc_rolling_rank(test_set[[col_p]], p_rank_window)
  
  test_set$filter_mp <- 1.0
  
  # 崩盘保护 (M < 20%)
  mask_crash <- test_set$rank_m < 0.20
  test_set$filter_mp[mask_crash] <- 0.0
  
  # 震荡观察 (20% <= M < 70%)
  mask_chop <- (test_set$rank_m >= 0.20) & (test_set$rank_m < 0.70)
  test_set$filter_mp[mask_chop] <- 0.5
  
  # 估值泡沫 (P > 90%)
  mask_bubble <- test_set$rank_p > 0.90
  test_set$filter_mp[mask_bubble & !mask_crash] <- 0.5
  
  # (M >= 66% & P < 70%) -> 加杠杆
  mask_boost <- (test_set$rank_m >= 0.66) & (test_set$rank_p < 0.70)
  test_set$filter_mp[mask_boost] <- 2.0
  
  # --- 风控与合成 ---
  vol_zoo <- rollapply(test_set$market_forward_excess_returns, 
                       width = 20, FUN = sd, fill = NA, align = "right") * sqrt(252)
  current_vol <- as.numeric(vol_zoo)
  current_vol[is.na(current_vol) | current_vol <= 0] <- p_target_vol
  test_set$regime_vol <- p_target_vol / current_vol
  
  # 合成最终仓位
  raw_pos <- test_set$signal_rf * test_set$filter_mp * test_set$regime_vol
  test_set$position <- pmin(raw_pos, p_max_leverage)
  test_set$position[is.na(test_set$position)] <- 0
  
  # --- 绩效统计 (含交易成本) ---
  pos_change <- diff(c(0, test_set$position))
  turnover_cost <- abs(pos_change) * p_trans_cost
  
  test_set$gross_return <- test_set$position * test_set$market_forward_excess_returns
  test_set$strategy_return <- test_set$gross_return - turnover_cost
  
  cat(sprintf("  - [Cost] 交易成本损耗: %.2f%%\n", sum(turnover_cost)*100))
  
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
  cat("\n----------- RF Sniper 自适应策略绩效 -----------\n")
  print(metrics_df)
  
  write.csv(metrics_df, file.path(config$output_dir, "rf_sniper_metrics.csv"), row.names = FALSE)
  
  plot_df <- data.frame(
    Time = 1:nrow(test_set),
    Market = cumprod(1 + test_set$market_forward_excess_returns),
    Strategy = cumprod(1 + test_set$strategy_return),
    Prob = test_set$pred_prob,
    Threshold = test_set$thresh_open
  )
  if("date_id" %in% names(test_set)) plot_df$Time <- test_set$date_id
  
  p1 <- ggplot(plot_df, aes(x = Time)) +
    geom_line(aes(y = Market, color = "Market"), alpha = 0.6) +
    geom_line(aes(y = Strategy, color = "Sniper (Net)"), linewidth = 1) +
    scale_color_manual(values = c("Market"="grey", "Sniper (Net)"="darkgreen")) +
    labs(title = "RF Sniper Adaptive Strategy",
         subtitle = paste("Sharpe:", round(res_strat['Sharpe'], 2)),
         y = "Net Value") +
    theme_minimal() +
    theme(legend.position = "top")
  
  tail_n <- min(500, nrow(plot_df))
  plot_df_tail <- tail(plot_df, tail_n)
  
  p2 <- ggplot(plot_df_tail, aes(x = Time)) +
    geom_line(aes(y = Prob, color = "RF Prob"), alpha = 0.5) +
    geom_line(aes(y = Threshold, color = "Dynamic Thresh"), linewidth = 0.8, linetype="dashed") +
    scale_color_manual(values = c("RF Prob"="blue", "Dynamic Thresh"="red")) +
    labs(title = "Sniper Threshold Monitor (Last 500 days)",
         y = "Probability / Threshold") +
    theme_minimal() +
    theme(legend.position = "top")
  
  ggsave(file.path(config$output_dir, "rf_sniper_equity.png"), p1, width=8, height=5)
  ggsave(file.path(config$output_dir, "rf_sniper_thresholds.png"), p2, width=8, height=4)
  
  return(list(
    results_df = test_set, 
    metrics = metrics_df, 
    plot = p1,
    platt_params = platt_params
  ))
}