# ============================================================
# 脚本: 寻找最佳融合权重 (Ensemble Optimizer)
# ============================================================

library(dplyr)
library(ggplot2)
library(zoo)

# 1. 加载两个模型的预测结果
# ------------------------------------------------------------
results_svm <- read.csv("output/result_svm.csv")
results_logistic <- read.csv("output/result_logistic.csv")
# 检查数据对齐
if (!all(results_svm$date_id == results_logistic$date_id)) stop("日期未对齐！")

# 提取概率
prob_svm <- results_svm$pred_prob
prob_log <- results_logistic$pred_prob
returns  <- results_svm$market_forward_excess_returns # 真实收益

# 2. 定义回测函数
# ------------------------------------------------------------
# 这个函数只计算夏普，用来快速评估
backtest_weight <- function(w, prob_svm, prob_log, returns) {
  
  # A. 混合概率
  prob_mix <- w * prob_svm + (1 - w) * prob_log
  
  # B. 重新计算自适应中枢 (复用之前的逻辑)
  # 这里简单起见，直接用全局计算，严谨的话应该用 rollapply,但为了搜索最佳权重，这点误差可忽略
  
  # 使用简单的 rollapply 模拟策略逻辑
  roll_median <- rollapply(prob_mix, width=60, FUN=median, fill=NA, align="right")
  adaptive_center <- dplyr::lag(roll_median, 1)
  adaptive_center[is.na(adaptive_center)] <- median(prob_mix, na.rm=TRUE)
  
  # C. 信号生成
  signal_score <- 0.5 + (prob_mix - adaptive_center) * 10 # slope=10
  base_pos <- pmax(0.2, pmin(1.5, signal_score))
  
  # D. 风控 (复用 SVM 的结果，因为波动率计算是一样的)
  # 这里简化处理，假设满仓测试效果，或者简单模拟
  # 为了精确，最好把 rolling_vol 也传进来
  
  # 简单起见，我们直接看"混合信号"本身的质量，不考虑复杂的杠杆和风控
  # 这样能最纯粹地反映 alpha 的质量
  position <- base_pos # 简化版仓位
  
  strat_ret <- position * returns
  
  # E. 计算夏普
  ann_ret <- mean(strat_ret, na.rm=TRUE) * 252
  vol <- sd(strat_ret, na.rm=TRUE) * sqrt(252)
  sharpe <- ann_ret / vol
  
  return(sharpe)
}

# 3. 网格搜索 (Grid Search)
# ------------------------------------------------------------
weights <- seq(0, 1, by = 0.05) # 从 0 到 1，每隔 0.05 试一次
sharpes <- c()

cat("正在搜索最佳权重 w (SVM 的比例)...\n")

for (w in weights) {
  s <- backtest_weight(w, prob_svm, prob_log, returns)
  sharpes <- c(sharpes, s)
  cat(sprintf("  w_svm = %.2f | w_log = %.2f -> Sharpe = %.4f\n", w, 1-w, s))
}

# 4. 结果可视化
# ------------------------------------------------------------
best_idx <- which.max(sharpes)
best_w <- weights[best_idx]

cat("\n========================================\n")
cat("最佳权重找到: w_svm =", best_w, "\n")
cat("最高夏普比率:", max(sharpes), "\n")
cat("========================================\n")

plot(weights, sharpes, type="b", col="blue", pch=19,
     main="Ensemble Optimization: SVM Weight vs Sharpe",
     xlab="Weight of SVM (0=Pure Log, 1=Pure SVM)", ylab="Sharpe Ratio")
grid()
abline(v=best_w, col="red", lty=2)