# ============================================================
# 模块: 手写 Lasso 回归 (坐标下降法) [修复版: 支持NA处理]
# 文件路径: XGB/lasso_handmade.R
# ============================================================

# 0. 辅助函数: 均值填充 (Mean Imputation) - [升级版]
# ------------------------------------------------------------
# 原理: 用现有数据的平均值填充 NA，保持数据的分布中心不变。
# 对于 Lasso 这种线性模型，这是最稳健的做法。
algo_impute_na <- function(X) {
  # 确保是矩阵
  X <- as.matrix(X)
  n_features <- ncol(X)
  
  # 遍历每一列
  for (j in 1:n_features) {
    # 找到这一列里 NA 的位置
    na_idx <- is.na(X[, j])
    
    # 如果有 NA
    if (any(na_idx)) {
      # 计算该列非 NA 数据的均值
      col_mean <- mean(X[!na_idx, j], na.rm = TRUE)
      
      # 如果整列都是 NA (极端情况)，则填 0
      if (is.na(col_mean)) col_mean <- 0
      
      # 填补
      X[na_idx, j] <- col_mean
    }
  }
  return(X)
}

# 1. 软阈值函数
# ------------------------------------------------------------
soft_threshold <- function(rho, lambda) {
  if (rho > lambda) return(rho - lambda)
  if (rho < -lambda) return(rho + lambda)
  return(0)
}

# 2. Lasso 求解器 (Solver)
# ------------------------------------------------------------
algo_lasso_solver <- function(X, y, lambda, tol = 1e-4, max_iter = 1000) {
  n_samples <- nrow(X)
  n_features <- ncol(X)
  w <- rep(0, n_features)
  
  # 预计算分母 (X是标准化的，colSums(X^2) 约等于 n_samples)
  z <- colSums(X^2)
  # 防止除以0的保护
  z[z == 0] <- 1
  
  r <- y # 初始残差 (假设 w=0)
  
  for (iter in 1:max_iter) {
    w_old <- w
    max_change <- 0
    
    for (j in 1:n_features) {
      # 坐标下降更新
      r_partial_corr <- sum(X[, j] * r) + z[j] * w[j]
      w_new <- soft_threshold(r_partial_corr, lambda * n_samples) / z[j]
      
      if (abs(w_new - w[j]) > 1e-10) {
        r <- r - X[, j] * (w_new - w[j])
        change <- abs(w_new - w[j])
        if (change > max_change) max_change <- change
        w[j] <- w_new
      }
    }
    if (max_change < tol) break
  }
  return(w)
}

# 3. CV 寻找最佳 Lambda (包含数据清洗)
# ------------------------------------------------------------
algo_lasso_cv <- function(X_raw, y_raw, k_folds = 5, n_lambdas = 20) {
  
  # [修复点 1] 强制处理 NA
  # Lasso 的矩阵运算不能有 NA，这里临时填充为 0
  if (any(is.na(X_raw))) {
    # cat("  [Lasso] 检测到 NA，执行临时零填充...\n")
    X_raw <- algo_impute_na(X_raw)
  }
  
  # --- A. 数据标准化 ---
  X_mean <- colMeans(X_raw)
  X_sd <- apply(X_raw, 2, sd)
  X_sd[X_sd == 0] <- 1 # 防止除以0
  
  X_std <- scale(X_raw, center = X_mean, scale = X_sd)
  
  y_mean <- mean(y_raw)
  y_std <- y_raw - y_mean
  
  # --- B. 生成 Lambda 序列 ---
  # 计算相关性梯度
  grads <- abs(t(X_std) %*% y_std)
  
  # [修复点 2] Lambda 安全计算
  lambda_max <- max(grads, na.rm = TRUE) / nrow(X_std)
  
  # 防止 lambda_max 为 0 (导致 log(-Inf))
  if (lambda_max <= 1e-6) {
    warning("  [Lasso] 警告: 梯度极小，可能所有特征都无效。重置 lambda_max。")
    lambda_max <- 0.1 
  }
  
  lambda_min <- lambda_max * 0.0001
  lambdas <- exp(seq(log(lambda_max), log(lambda_min), length.out = n_lambdas))
  
  # --- C. K-Fold ---
  n <- nrow(X_raw)
  # 确保 fold_ids 长度匹配
  fold_ids <- sample(rep(1:k_folds, length.out = n))
  
  cv_errors <- matrix(0, nrow = n_lambdas, ncol = k_folds)
  
  for (k in 1:k_folds) {
    test_mask <- (fold_ids == k)
    
    X_tr <- X_std[!test_mask, , drop=FALSE]
    y_tr <- y_std[!test_mask]
    X_val <- X_std[test_mask, , drop=FALSE]
    y_val <- y_std[test_mask]
    
    # Warm start: 上一个 lambda 的解作为下一个的初始值
    w_current <- rep(0, ncol(X_tr))
    
    for (i in 1:n_lambdas) {
      lam <- lambdas[i]
      # 简化的 Solver 调用
      w_hat <- algo_lasso_solver(X_tr, y_tr, lam)
      
      preds <- X_val %*% w_hat
      cv_errors[i, k] <- mean((y_val - preds)^2)
    }
  }
  
  # --- D. 结果 ---
  avg_mse <- rowMeans(cv_errors, na.rm = TRUE)
  best_idx <- which.min(avg_mse)
  best_lambda <- lambdas[best_idx]
  
  cat(sprintf("  [Handmade Lasso] Lambda选定: %.5f (MSE: %.4f)\n", best_lambda, min(avg_mse)))
  
  # 全量训练
  final_w <- algo_lasso_solver(X_std, y_std, best_lambda)
  
  # 还原系数到原始尺度
  final_w_raw <- final_w / X_sd
  
  # 提取特征
  selected_indices <- which(abs(final_w) > 1e-10) # 容差判断非零
  selected_feats <- colnames(X_raw)[selected_indices]
  
  # [修复点 3] 兜底策略
  # 如果 Lasso 把所有特征都删光了，强制保留相关性最高的前5个
  if (length(selected_feats) == 0) {
    cat("  [Lasso] 警告: 所有系数归零。回退策略: 选择Top 5相关特征。\n")
    corrs <- abs(cor(X_raw, y_raw))
    top5_idx <- order(corrs, decreasing = TRUE)[1:5]
    selected_feats <- colnames(X_raw)[top5_idx]
  }
  
  return(list(
    best_lambda = best_lambda,
    selected_features = selected_feats
  ))
}