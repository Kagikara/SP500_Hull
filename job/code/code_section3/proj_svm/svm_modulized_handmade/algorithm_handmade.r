# ------------------------------------------------------------
#  Z-Score 标准化算法
# ------------------------------------------------------------
#' @param X 特征矩阵或数据框
#' @return list(mean, sd)
algo_zscore_fit <- function(X) {
  # 强制转换为矩阵
  X_mat <- as.matrix(X)
  
  # 1. 计算列均值
  mu <- colMeans(X_mat, na.rm = TRUE)
  
  # 2. 计算列标准差 (使用apply循环)
  sigma <- apply(X_mat, 2, sd, na.rm = TRUE)
  
  sigma[sigma == 0] <- 1
  
  return(list(mean = mu, sd = sigma))
}

#' [算法] 应用标准化转换
#' 数学原理: z_i = (x_i - mu) / sigma
#' @param X 需要转换的数据
#' @param params algo_zscore_fit 返回的参数列表
#' @return 标准化后的矩阵
algo_zscore_transform <- function(X, params) {
  X_mat <- as.matrix(X)
  mu <- params$mean
  sigma <- params$sd
  
  if (ncol(X_mat) != length(mu)) {
    stop(paste("维度不匹配: 数据列数", ncol(X_mat), "!= 参数列数", length(mu)))
  }
  X_scaled <- t((t(X_mat) - mu) / sigma)
  
  return(X_scaled)
}

#' [算法] 线性核矩阵计算
algo_kernel_linear <- function(X1, X2) {
  tcrossprod(as.matrix(X1), as.matrix(X2))
}

#' [算法] RBF 核矩阵计算
#' @param X1 矩阵 n x d
#' @param X2 矩阵 m x d
#' @param gamma 核参数
algo_kernel_rbf <- function(X1, X2, gamma) {
  X1 <- as.matrix(X1)
  X2 <- as.matrix(X2)
  
  n <- nrow(X1)
  m <- nrow(X2)
  
  sq_norm1 <- rowSums(X1^2)
  sq_norm2 <- rowSums(X2^2)
  dot_prod <- tcrossprod(X1, X2)
  
  dist_sq <- outer(sq_norm1, sq_norm2, "+") - 2 * dot_prod
  
  K <- exp(-gamma * dist_sq)
  return(K)
}

#' [算法]SMO训练SVM
#' @param kernel_type "linear" 或 "rbf"
#' @param gamma RBF核需要此参数，Linear核忽略
algo_svm_train_smo <- function(X, y, C, kernel_type="linear", gamma=NULL, max_iter=100, tol=1e-3) {
  
  X_mat <- as.matrix(X)
  y_vec <- as.numeric(as.character(y)) 
  if (all(y_vec %in% c(0, 1))) y_vec[y_vec == 0] <- -1
  
  n_samples <- nrow(X_mat)
  alphas <- numeric(n_samples)
  b <- 0
  
  # --------------------------------------------------------
  # 根据 kernel_type 预计算核矩阵 K
  # --------------------------------------------------------
  if (kernel_type == "rbf") {
    if(is.null(gamma)) stop("RBF核必须提供 gamma 参数")
    K <- algo_kernel_rbf(X_mat, X_mat, gamma)
  } else if (kernel_type == "linear") {
    K <- algo_kernel_linear(X_mat, X_mat)
  } else {
    stop(paste("未知的 kernel_type:", kernel_type))
  }
  
  passes <- 0 
  iter_count <- 0
  
  cat(sprintf("  [SMO] Start: Kernel=%s, C=%.4f, Samples=%d\n", kernel_type, C, n_samples))
  
  while (passes < 10 && iter_count < max_iter) {
    num_changed_alphas <- 0
    iter_count <- iter_count + 1
    
    for (i in 1:n_samples) {
      f_xi <- sum(alphas * y_vec * K[, i]) + b
      E_i <- f_xi - y_vec[i]
      
      if ((y_vec[i] * E_i < -tol && alphas[i] < C) || 
          (y_vec[i] * E_i > tol && alphas[i] > 0)) {
        
        j <- sample(setdiff(1:n_samples, i), 1)
        f_xj <- sum(alphas * y_vec * K[, j]) + b
        E_j <- f_xj - y_vec[j]
        
        alpha_i_old <- alphas[i]
        alpha_j_old <- alphas[j]
        
        if (y_vec[i] != y_vec[j]) {
          L <- max(0, alpha_j_old - alpha_i_old)
          H <- min(C, C + alpha_j_old - alpha_i_old)
        } else {
          L <- max(0, alpha_i_old + alpha_j_old - C)
          H <- min(C, alpha_i_old + alpha_j_old)
        }
        
        if (L == H) next
        
        eta <- 2 * K[i, j] - K[i, i] - K[j, j]
        if (eta >= 0) next
        
        alphas[j] <- alphas[j] - (y_vec[j] * (E_i - E_j)) / eta
        
        if (alphas[j] > H) alphas[j] <- H
        if (alphas[j] < L) alphas[j] <- L
        
        if (abs(alphas[j] - alpha_j_old) < 1e-5) next
        
        alphas[i] <- alphas[i] + y_vec[i] * y_vec[j] * (alpha_j_old - alphas[j])
        
        b1 <- b - E_i - y_vec[i] * (alphas[i] - alpha_i_old) * K[i, i] - 
              y_vec[j] * (alphas[j] - alpha_j_old) * K[i, j]
        b2 <- b - E_j - y_vec[i] * (alphas[i] - alpha_i_old) * K[i, j] - 
              y_vec[j] * (alphas[j] - alpha_j_old) * K[j, j]
        
        if (0 < alphas[i] && alphas[i] < C) { b <- b1 } 
        else if (0 < alphas[j] && alphas[j] < C) { b <- b2 } 
        else { b <- (b1 + b2) / 2 }
        
        num_changed_alphas <- num_changed_alphas + 1
      }
    }
    
    if (num_changed_alphas == 0) passes <- passes + 1 else passes <- 0
  }
  
  sv_idx <- which(alphas > 1e-5)
  
  return(list(
    alphas = alphas[sv_idx],
    sv_indices = sv_idx,
    sv_X = X_mat[sv_idx, , drop=FALSE],
    sv_y = y_vec[sv_idx],
    b = b,
    gamma = gamma,
    kernel = kernel_type
  ))
}

#' [算法]SVM预测
#' 根据模型中的 kernel 类型自动选择计算方式
algo_svm_predict_score <- function(model, X_new) {
  X_new <- as.matrix(X_new)
  
  if (model$kernel == "rbf") {
    K_new <- algo_kernel_rbf(X_new, model$sv_X, model$gamma)
  } else if (model$kernel == "linear") {
    K_new <- algo_kernel_linear(X_new, model$sv_X)
  } else {
    stop("模型包含未知的 kernel 类型")
  }
  
  weights <- model$alphas * model$sv_y
  decision_values <- (K_new %*% weights) + model$b
  
  return(as.vector(decision_values))
}

########### 还没有输出概率 ###############

# ============================================================
# 3. 概率校准
# ============================================================

#' [算法] 使用牛顿法求解 Sigmoid 参数
#' @param scores SVM 输出的决策值向量 f(x)
#' @param y 真实标签向量 (0 或 1)
#' @return list(A, B)
algo_platt_fit <- function(scores, y, max_iter = 100, tol = 1e-5) {
  # Platt 对标签进行平滑处理以避免过拟合
  prior0 <- sum(y == 0)
  prior1 <- sum(y == 1)
  
  hi_target <- (prior1 + 1.0) / (prior1 + 2.0)
  lo_target <- 1 / (prior0 + 2.0)
  
  t <- ifelse(y == 1, hi_target, lo_target)
  
  A <- 0.0
  B <- log((prior0 + 1.0) / (prior1 + 1.0))
  
  # 牛顿法迭代
  for (it in 1:max_iter) {
    f_ap_b <- scores * A + B
    
    p <- ifelse(f_ap_b >= 0, 
                exp(-f_ap_b) / (1.0 + exp(-f_ap_b)), 
                1.0 / (1.0 + exp(f_ap_b)))
    
    d <- p - t
    h <- p * (1.0 - p)
    sigma <- 1e-12 # 防止除零
    h <- pmax(h, sigma)
    
    grad_A <- sum(scores * d)
    grad_B <- sum(d)
    hess_AA <- sum(scores * scores * h)
    hess_BB <- sum(h)
    hess_AB <- sum(scores * h)
    
    det <- hess_AA * hess_BB - hess_AB * hess_AB
    
    if (abs(det) < 1e-12) break
    
    dA <- (hess_BB * grad_A - hess_AB * grad_B) / det
    dB <- (hess_AA * grad_B - hess_AB * grad_A) / det
    
    old_A <- A
    old_B <- B
    
    A <- A + dA
    B <- B + dB
    
    if (abs(A - old_A) < tol && abs(B - old_B) < tol) break
  }
  
  return(list(A = A, B = B))
}

#' [算法] Sigmoid 预测
#' @param scores 决策值
#' @param params Platt Scaling 参数 (A, B)
#' @return 概率值
algo_platt_predict <- function(scores, params) {
  f_ap_b <- scores * params$A + params$B
  
  # Sigmoid 函数
  p <- ifelse(f_ap_b >= 0, 
              exp(-f_ap_b) / (1.0 + exp(-f_ap_b)), 
              1.0 / (1.0 + exp(f_ap_b)))
  return(p)
}

# ============================================================
# 4. Logistic 回归核心算法
# ============================================================

#' [算法] Sigmoid 函数
algo_sigmoid <- function(z) {
  1 / (1 + exp(-z))
}

#' [算法] Logistic 回归训练 (梯度下降法)
#' @param X 特征矩阵
#' @param y 标签 (0/1)
#' @param learning_rate 学习率
#' @param lambda 正则化系数 (L2)
#' @param max_iter 最大迭代次数
algo_logistic_train_gd <- function(X, y, learning_rate = 0.01, lambda = 0.1, max_iter = 1000) {
  
  X_mat <- as.matrix(X)
  y_vec <- as.numeric(as.character(y))
  
  m <- nrow(X_mat)
  n <- ncol(X_mat)
  
  weights <- rep(0, n)
  bias <- 0
  for (i in 1:max_iter) {
    z <- (X_mat %*% weights) + bias
    p <- algo_sigmoid(z)
    error <- p - y_vec
    dw <- (1/m) * (t(X_mat) %*% error) + (lambda/m) * weights
    db <- (1/m) * sum(error)
    
    weights <- weights - learning_rate * dw
    bias <- bias - learning_rate * db
  }
  
  return(list(
    weights = weights,
    bias = bias,
    model_type = "logistic"
  ))
}

#' [算法] Logistic 预测
#' @return list(scores, probs)
algo_logistic_predict <- function(model, X) {
  X_mat <- as.matrix(X)
  
  z <- (X_mat %*% model$weights) + model$bias
  
  probs <- algo_sigmoid(z)
  
  return(list(scores = as.vector(z), probs = as.vector(probs)))
}


# ============================================================
# 5. RF
# 组件: CART 决策树, Gini 纯度, Bootstrap, Bagging
# ============================================================

# ------------------------------------------------------------
# A. 基础函数
# ------------------------------------------------------------

#' [算法] 计算 Gini 不纯度
#' 公式: 1 - sum(p_i^2)
algo_cart_gini <- function(y) {
  if (length(y) == 0) return(0)
  props <- prop.table(table(y))
  1 - sum(props^2)
}

#' [算法] 寻找最佳切分点
#' @param X 特征矩阵
#' @param y 标签向量
#' @param mtry 随机采样的特征数量
#' @return list(feature_idx, threshold, gain) 或 NULL
algo_cart_find_split <- function(X, y, mtry) {
  
  n_samples <- nrow(X)
  n_features <- ncol(X)
  current_gini <- algo_cart_gini(y)
  
  best_gain <- 0
  best_split <- NULL
  
  # 随机选择 mtry 个特征的索引
  feat_indices <- sample(1:n_features, min(mtry, n_features))
  
  # 遍历选中的特征
  for (feat_idx in feat_indices) {
    x_col <- X[, feat_idx]
    
    if (length(unique(x_col)) <= 1) next
    
    ord <- order(x_col)
    x_sorted <- x_col[ord]
    y_sorted <- y[ord]

    is_change <- c(diff(x_sorted) != 0, FALSE) # 最后一位补FALSE
    if (!any(is_change)) next
    
    split_indices <- which(is_change)
    
    # ------------------------------------------------------
    # 计算所有切分点的 Gini
    # ------------------------------------------------------
    y_numeric <- as.numeric(y_sorted) - 1
    
    n_left <- split_indices
    n_right <- n_samples - n_left
    cum_sum <- cumsum(y_numeric)
    p1_left_count <- cum_sum[split_indices]
    total_p1 <- sum(y_numeric)
    p1_right_count <- total_p1 - p1_left_count
    
    gini_left <- 1 - ( (p1_left_count/n_left)^2 + ((n_left-p1_left_count)/n_left)^2 )
    gini_right <- 1 - ( (p1_right_count/n_right)^2 + ((n_right-p1_right_count)/n_right)^2 )
    
    gini_split <- (n_left / n_samples) * gini_left + (n_right / n_samples) * gini_right
    
    gains <- current_gini - gini_split
    
    max_g_idx <- which.max(gains)
    max_g <- gains[max_g_idx]
    
    if (max_g > best_gain) {
      best_gain <- max_g
      idx_in_sorted <- split_indices[max_g_idx]
      threshold <- (x_sorted[idx_in_sorted] + x_sorted[idx_in_sorted+1]) / 2
      
      best_split <- list(
        feature_idx = feat_idx,
        threshold = threshold,
        gain = best_gain
      )
    }
  }
  
  return(best_split)
}

# ------------------------------------------------------------
# B. 树构建函数
# ------------------------------------------------------------

#' [算法] 递归构建决策树
#' @param current_depth 当前深度
#' @param max_depth 最大深度限制
#' @param min_node_size 叶节点最小样本数
algo_cart_build_tree <- function(X, y, current_depth, max_depth, min_node_size, mtry) {
  
  # 1. 检查停止条件
  if (current_depth >= max_depth || 
      nrow(X) < min_node_size || 
      length(unique(y)) == 1) {
    
    # 返回叶节点
    prob_1 <- mean(as.numeric(y) == 1)
    return(list(type = "leaf", prob = prob_1, samples = nrow(X)))
  }
  
  split <- algo_cart_find_split(X, y, mtry)
  
  if (is.null(split)) {
    prob_1 <- mean(as.numeric(y) == 1)
    return(list(type = "leaf", prob = prob_1, samples = nrow(X)))
  }

  feat_col <- X[, split$feature_idx]
  left_mask <- feat_col <= split$threshold
  
  X_left <- X[left_mask, , drop=FALSE]
  y_left <- y[left_mask]
  X_right <- X[!left_mask, , drop=FALSE]
  y_right <- y[!left_mask]

  left_node <- algo_cart_build_tree(X_left, y_left, current_depth + 1, max_depth, min_node_size, mtry)
  right_node <- algo_cart_build_tree(X_right, y_right, current_depth + 1, max_depth, min_node_size, mtry)
  
  return(list(
    type = "split",
    feature_idx = split$feature_idx,
    threshold = split$threshold,
    left = left_node,
    right = right_node,
    gain = split$gain
  ))
}

# ------------------------------------------------------------
# C. 预测函数
# ------------------------------------------------------------

#' [算法] 单样本预测
algo_cart_predict_sample <- function(tree, sample_vec) {
  node <- tree
  while (node$type == "split") {
    val <- sample_vec[node$feature_idx]
    if (val <= node$threshold) {
      node <- node$left
    } else {
      node <- node$right
    }
  }
  return(node$prob)
}

#' [算法] 整树预测 (批量)
algo_cart_predict_batch <- function(tree, X) {
  probs <- apply(X, 1, function(row) algo_cart_predict_sample(tree, row))
  return(probs)
}

# ------------------------------------------------------------
# D. Bagging
# ------------------------------------------------------------

#' [算法] RF训练
algo_rf_train_custom <- function(X, y, ntree, mtry, max_depth = 5, min_node_size = 10) {
  
  n_samples <- nrow(X)
  forest <- list()
  
  # 循环建立 ntree 棵树
  for (i in 1:ntree) {
    # Bootstrap
    boot_idx <- sample(1:n_samples, n_samples, replace = TRUE)
    X_boot <- X[boot_idx, , drop=FALSE]
    y_boot <- y[boot_idx]
    
    # 构建树
    tree <- algo_cart_build_tree(X_boot, y_boot, 
                                 current_depth = 0, 
                                 max_depth = max_depth, 
                                 min_node_size = min_node_size, 
                                 mtry = mtry)
    
    forest[[i]] <- tree
  }
  
  return(forest)
}