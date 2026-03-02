library(tidyverse)
library(ggplot2)
setwd("C:/Users/kagirasu/works_H/ml_HULL/job/code")
data_P <- read.csv("code_section3/proj_svm/data/train.csv", stringsAsFactors = FALSE)

p_colindex <- c(58:70)
P_star_signal <- rowMeans(data_P[, p_colindex], na.rm = TRUE)

R_m_return <- data_P$market_forward_excess_returns

P_star_filtered <- P_star_signal[-(1:1006)]
R_m_filtered <- R_m_return[-(1:1006)]

n_target <- floor(length(P_star_filtered) * 0.20)
sorted_indices <- order(P_star_filtered)
label <- rep(FALSE, length(P_star_filtered))
label[sorted_indices[1:n_target]] <- TRUE

R_Q1 <- ifelse(label, R_m_filtered, 0)
R_P <- R_m_filtered

# ============================================================================
# 步骤 1: 定义MDD计算函数
# ============================================================================

#' 计算最大回撤比 (Maximum Drawdown)
#' 
#' @param returns 收益率序列（数值向量）
#' @return 最大回撤比（负数，例如-0.25表示25%的回撤）
calculate_mdd <- function(returns) {
  # 检查输入
  if (length(returns) == 0 || all(is.na(returns))) {
    return(NA)
  }
  
  # 计算净值曲线（累积收益）
  equity_curve <- cumprod(1 + returns)
  
  # 计算历史峰值
  running_max <- cummax(equity_curve)
  
  # 计算回撤序列
  drawdown <- (equity_curve - running_max) / running_max
  
  # 返回最大回撤（最小值）
  mdd <- min(drawdown, na.rm = TRUE)
  
  return(mdd)
}

# ============================================================================
# 步骤 2: 定义块Bootstrap函数
# ============================================================================

#' 循环块Bootstrap重采样
#' 
#' @param data1 第一个收益序列
#' @param data2 第二个收益序列
#' @param block_length 块长度
#' @return 列表，包含重采样后的两个序列
block_bootstrap_resample <- function(data1, data2, block_length) {
  N <- length(data1)
  
  # 计算需要的块数量
  num_blocks <- ceiling(N / block_length)
  
  # 初始化新序列
  new_data1 <- numeric(0)
  new_data2 <- numeric(0)
  
  # 循环抽取数据块
  for (j in 1:num_blocks) {
    # 随机选择起始点
    start_idx <- sample(1:N, 1)
    
    # 计算块的索引（处理环绕）
    block_indices <- ((start_idx - 1) + 0:(block_length - 1)) %% N + 1
    
    # 抽取数据块并追加
    new_data1 <- c(new_data1, data1[block_indices])
    new_data2 <- c(new_data2, data2[block_indices])
  }
  
  # 截断到原始长度
  new_data1 <- new_data1[1:N]
  new_data2 <- new_data2[1:N]
  
  return(list(data1 = new_data1, data2 = new_data2))
}

# ============================================================================
# 步骤 3: 主检验函数
# ============================================================================

#' MDD差异的块Bootstrap检验
#' 
#' @param R_Q1 策略1的收益序列（例如价值择时策略）
#' @param R_P 策略2的收益序列（例如被动持有策略）
#' @param B Bootstrap重复次数（默认5000）
#' @param block_length 块长度（默认10）
#' @param conf_level 置信水平（默认0.95）
#' @param seed 随机种子（默认NULL）
#' @return 包含检验结果的列表
mdd_bootstrap_test <- function(R_Q1, R_P, B = 5000, block_length = 10, 
                                conf_level = 0.95, seed = NULL) {
  
  # 设置随机种子（可重复性）
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  # -------------------------------------------------------------------------
  # 步骤 1: 计算观测到的真实差异
  # -------------------------------------------------------------------------
  cat("步骤 1: 计算观测到的真实MDD差异...\n")
  
  mdd_q1_observed <- calculate_mdd(R_Q1)
  mdd_p_observed <- calculate_mdd(R_P)
  observed_difference <- mdd_q1_observed - mdd_p_observed
  
  cat(sprintf("  策略1 (Q1) MDD: %.4f (%.2f%%)\n", 
              mdd_q1_observed, mdd_q1_observed * 100))
  cat(sprintf("  策略2 (P)  MDD: %.4f (%.2f%%)\n", 
              mdd_p_observed, mdd_p_observed * 100))
  cat(sprintf("  观测差异: %.4f (%.2f%%)\n\n", 
              observed_difference, observed_difference * 100))
  
  # -------------------------------------------------------------------------
  # 步骤 2 & 3: Bootstrap模拟循环
  # -------------------------------------------------------------------------
  cat(sprintf("步骤 2-3: 执行Bootstrap模拟 (B=%d, 块长度=%d)...\n", B, block_length))
  
  # 初始化结果向量
  bootstrap_results <- numeric(B)
  
  # 进度条
  pb <- txtProgressBar(min = 0, max = B, style = 3)
  
  # Bootstrap循环
  for (i in 1:B) {
    # 重采样
    resampled <- block_bootstrap_resample(R_Q1, R_P, block_length)
    
    # 计算模拟的MDD
    mdd_q1_sim <- calculate_mdd(resampled$data1)
    mdd_p_sim <- calculate_mdd(resampled$data2)
    
    # 计算并存储差异
    bootstrap_results[i] <- mdd_q1_sim - mdd_p_sim
    
    # 更新进度条
    setTxtProgressBar(pb, i)
  }
  close(pb)
  
  cat("\nBootstrap模拟完成!\n\n")
  
  # -------------------------------------------------------------------------
  # 步骤 4: 构建置信区间
  # -------------------------------------------------------------------------
  cat("步骤 4: 构建置信区间...\n")
  
  # 计算分位数
  alpha <- 1 - conf_level
  lower_bound <- quantile(bootstrap_results, alpha / 2, na.rm = TRUE)
  upper_bound <- quantile(bootstrap_results, 1 - alpha / 2, na.rm = TRUE)
  
  cat(sprintf("  %.1f%% 置信区间: [%.4f, %.4f]\n", 
              conf_level * 100, lower_bound, upper_bound))
  cat(sprintf("  即: [%.2f%%, %.2f%%]\n\n", 
              lower_bound * 100, upper_bound * 100))
  
  # -------------------------------------------------------------------------
  # 步骤 5: 统计检验结论
  # -------------------------------------------------------------------------
  cat("步骤 5: 统计检验结论\n")
  cat("=" , rep("=", 70), "\n", sep = "")
  
  # 检验零假设
  reject_null <- !(0 >= lower_bound && 0 <= upper_bound)
  
  if (reject_null) {
    cat(sprintf("✓ 拒绝H0: 0不在置信区间内\n"))
    cat(sprintf("  结论: 有%.0f%%的置信度认为,两个策略的MDD存在显著差异。\n", 
                conf_level * 100))
    if (observed_difference > 0) {
      cat("  策略1 (Q1) 的MDD显著优于策略2 (P)。\n")
    } else {
      cat("  策略2 (P) 的MDD显著优于策略1 (Q1)。\n")
    }
  } else {
    cat(sprintf("✗ 未能拒绝H0: 0包含在置信区间内\n"))
    cat(sprintf("  结论: 两个策略的MDD差异在统计上不显著(α=%.2f)。\n", alpha))
  }
  cat("=" , rep("=", 70), "\n\n", sep = "")
  
  # -------------------------------------------------------------------------
  # 返回结果
  # -------------------------------------------------------------------------
  results <- list(
    # 观测值
    mdd_q1_observed = mdd_q1_observed,
    mdd_p_observed = mdd_p_observed,
    observed_difference = observed_difference,
    
    # Bootstrap结果
    bootstrap_results = bootstrap_results,
    
    # 置信区间
    conf_level = conf_level,
    lower_bound = as.numeric(lower_bound),
    upper_bound = as.numeric(upper_bound),
    
    # 检验结论
    reject_null = reject_null,
    
    # 参数
    B = B,
    block_length = block_length,
    N = length(R_Q1)
  )
  
  class(results) <- "mdd_bootstrap_test"
  return(results)
}

# ============================================================================
# 可视化函数
# ============================================================================

#' 绘制Bootstrap结果
#' 
#' @param test_result mdd_bootstrap_test函数的返回结果
plot_bootstrap_results <- function(test_result) {
  
  df <- data.frame(difference = test_result$bootstrap_results)
  
  p <- ggplot(df, aes(x = difference)) +
    geom_histogram(aes(y = after_stat(density)), bins = 50, 
                   fill = "skyblue", color = "black", alpha = 0.7) +
    geom_density(color = "darkblue", linewidth = 1) +
    geom_vline(xintercept = test_result$observed_difference, 
               color = "red", linetype = "dashed", linewidth = 1) +
    geom_vline(xintercept = 0, 
               color = "black", linetype = "solid", linewidth = 0.5) +
    geom_vline(xintercept = test_result$lower_bound, 
               color = "orange", linetype = "dashed", linewidth = 1) +
    geom_vline(xintercept = test_result$upper_bound, 
               color = "orange", linetype = "dashed", linewidth = 1) +
    labs(
      title = "MDD差异的Bootstrap分布",
      subtitle = sprintf("B=%d, 块长度=%d, %.0f%%置信区间", 
                        test_result$B, test_result$block_length, 
                        test_result$conf_level * 100),
      x = "MDD差异 (Q1 - P)",
      y = "密度"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5)
    ) +
    annotate("text", x = test_result$observed_difference, 
             y = Inf, label = "观测值", 
             vjust = 2, color = "red", fontface = "bold")
  
  print(p)
  
  return(p)
}

# ============================================================================
# 示例使用
# ============================================================================

test_result <- mdd_bootstrap_test(
  R_Q1 = R_Q1,
  R_P = R_P,
  B = 5000,              # Bootstrap重复次数
  block_length = 10,     # 块长度
  conf_level = 0.95,     # 置信水平
  seed = 42              # 随机种子
)

# 绘制结果
plot_bootstrap_results(test_result)

# 查看详细结果
cat("\n详细结果摘要:\n")
cat("=" , rep("=", 70), "\n", sep = "")
cat(sprintf("样本量: %d\n", test_result$N))
cat(sprintf("Bootstrap次数: %d\n", test_result$B))
cat(sprintf("块长度: %d\n", test_result$block_length))
cat(sprintf("置信水平: %.1f%%\n", test_result$conf_level * 100))
cat(sprintf("\n观测差异均值: %.4f\n", mean(test_result$bootstrap_results)))
cat(sprintf("观测差异标准差: %.4f\n", sd(test_result$bootstrap_results)))
cat("=" , rep("=", 70), "\n", sep = "")
