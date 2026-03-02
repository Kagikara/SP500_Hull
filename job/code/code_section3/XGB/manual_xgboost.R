sigmoid <- function(z) {
  z <- pmin(pmax(z, -30), 30)
  return(1 / (1 + exp(-z)))
}

calc_log_loss <- function(y_true, y_prob) {
  eps <- 1e-15
  y_prob <- pmin(pmax(y_prob, eps), 1 - eps)
  return(-mean(y_true * log(y_prob) + (1 - y_true) * log(1 - y_prob)))
}

calc_leaf_weight <- function(g, h, lambda_reg) {
  return(-sum(g) / (sum(h) + lambda_reg))
}

calc_structure_score <- function(g, h, lambda_reg) {
  return(sum(g)^2 / (sum(h) + lambda_reg))
}

# ============================================================
#  build_tree
# ============================================================
build_tree <- function(X, g, h, depth, params) {
  

  if (depth >= params$max_depth || nrow(X) <= 1) {
    leaf_weight <- calc_leaf_weight(g, h, params$lambda_reg)
    return(list(
      node = list(is_leaf = TRUE, weight = leaf_weight),
      stats = list()
    ))
  }
  
  score_p <- calc_structure_score(g, h, params$lambda_reg)
  
  best_gain <- -Inf
  best_feature <- NULL
  best_val <- NULL
  best_sets <- NULL
  
  n_features <- ncol(X)
  
  for (feat_idx in 1:n_features) {
    vals <- sort(unique(X[, feat_idx]), na.last = NA)
    
    if (length(vals) == 0) next
    
    col_data <- X[, feat_idx]
    
    for (val in vals) {
      mask_left <- !is.na(col_data) & (col_data <= val)
      
      if (!any(mask_left) || all(mask_left)) next
      
      g_L <- g[mask_left]; h_L <- h[mask_left]
      g_R <- g[!mask_left]; h_R <- h[!mask_left]
      
      score_L <- calc_structure_score(g_L, h_L, params$lambda_reg)
      score_R <- calc_structure_score(g_R, h_R, params$lambda_reg)
      
      gain <- 0.5 * (score_L + score_R - score_p) - params$gamma
      
      if (is.na(gain)) gain <- -Inf
      
      if (gain > best_gain) {
        best_gain <- gain
        best_feature <- feat_idx
        best_val <- val
        best_sets <- mask_left 
      }
    }
  }
  
  if (best_gain > 0) {
    mask_left <- best_sets
    mask_right <- !mask_left
    
    left_res <- build_tree(X[mask_left, , drop=FALSE], g[mask_left], h[mask_left], depth + 1, params)
    right_res <- build_tree(X[mask_right, , drop=FALSE], g[mask_right], h[mask_right], depth + 1, params)
    
    node <- list(
      is_leaf = FALSE,
      split_feature = best_feature, 
      split_val = best_val,
      left = left_res$node,
      right = right_res$node
    )
    
    curr_stats <- list(
      gain = setNames(best_gain, best_feature),
      cover = setNames(sum(h), best_feature),
      weight = setNames(1, best_feature)
    )
    
    all_stats <- list(
      gain = c(curr_stats$gain, left_res$stats$gain, right_res$stats$gain),
      cover = c(curr_stats$cover, left_res$stats$cover, right_res$stats$cover),
      weight = c(curr_stats$weight, left_res$stats$weight, right_res$stats$weight)
    )
    
    return(list(node = node, stats = all_stats))
    
  } else {
    leaf_weight <- calc_leaf_weight(g, h, params$lambda_reg)
    return(list(
      node = list(is_leaf = TRUE, weight = leaf_weight),
      stats = list()
    ))
  }
}

# ============================================================
# predict_one_tree
# ============================================================
predict_one_tree <- function(tree_root, X) {
  predict_row <- function(row, node) {
    if (node$is_leaf) {
      return(node$weight)
    }
    
    val <- row[node$split_feature]
    
    if (is.na(val)) {
      return(predict_row(row, node$right))
    }
    
    if (val <= node$split_val) {
      return(predict_row(row, node$left))
    } else {
      return(predict_row(row, node$right))
    }
  }
  return(apply(X, 1, predict_row, node = tree_root))
}

# ============================================================
# manual_xgboost
# ============================================================

xgb_train_manual <- function(X, y, params, nrounds, eval_set = NULL, verbose = FALSE, print_prefix = "") {
  n_samples <- nrow(X)
  
  # 初始化 F0
  init_pred <- 0.0
  y_pred <- rep(init_pred, n_samples)
  
  trees <- list()
  history <- list(train_loss = c(), val_loss = c())
  
  # 初始化特征重要性累加
  feat_imp <- list(gain = list(), cover = list(), weight = list())
  
  # 验证集初始化
  if (!is.null(eval_set)) {
    X_val <- eval_set$data
    y_val <- eval_set$label
    y_pred_val <- rep(init_pred, length(y_val))
  }
  
  cat(sprintf("%s 开始训练: n_samples=%d, n_features=%d, nrounds=%d\n", 
              print_prefix, nrow(X), ncol(X), nrounds))
  
  total_start_time <- Sys.time()
  
  for (t in 1:nrounds) {
    iter_start_time <- Sys.time()
    
    p <- sigmoid(y_pred)
    g <- p - y
    h <- p * (1 - p)
    
    # 构建树
    tree_res <- build_tree(X, g, h, depth = 0, params = params)
    
    # 保存树
    trees[[t]] <- tree_res$node
    
    # 更新训练集预测
    update_vals <- predict_one_tree(tree_res$node, X)
    y_pred <- y_pred + params$eta * update_vals
    
    # 记录 Loss
    train_loss <- calc_log_loss(y, sigmoid(y_pred))
    history$train_loss <- c(history$train_loss, train_loss)
    
    # 处理特征重要性
    new_stats <- tree_res$stats
    if (length(new_stats) > 0) {
      for (type in c("gain", "cover", "weight")) {
        vec <- new_stats[[type]]
        for (feat_idx in names(vec)) {
          val <- vec[[feat_idx]]
          current <- feat_imp[[type]][[feat_idx]]
          if (is.null(current)) current <- 0
          feat_imp[[type]][[feat_idx]] <- current + val
        }
      }
    }
    
    # 计算本轮耗时
    iter_end_time <- Sys.time()
    iter_duration <- as.numeric(difftime(iter_end_time, iter_start_time, units = "secs"))
    
    # 验证集逻辑与打印
    log_msg <- sprintf("%s Iter %d/%d | Time: %.1fs | Train Loss: %.5f", 
                       print_prefix, t, nrounds, iter_duration, train_loss)
    
    if (!is.null(eval_set)) {
      update_val_vals <- predict_one_tree(tree_res$node, X_val)
      y_pred_val <- y_pred_val + params$eta * update_val_vals
      val_loss <- calc_log_loss(y_val, sigmoid(y_pred_val))
      history$val_loss <- c(history$val_loss, val_loss)
      
      log_msg <- paste0(log_msg, sprintf(" | Val Loss: %.5f", val_loss))
    }
    
    if (verbose || t %% 1 == 0) {
      cat(paste0(log_msg, "\n"))
      flush.console()
    }
  }
  
  total_duration <- as.numeric(difftime(Sys.time(), total_start_time, units = "mins"))
  cat(sprintf("%s 训练结束. 总耗时: %.2f 分钟\n", print_prefix, total_duration))
  
  model <- list(
    trees = trees,
    params = params,
    init_pred = init_pred,
    feature_importance_raw = feat_imp,
    history = history
  )
  return(model)
}

# 主预测函数
xgb_predict_manual <- function(model, X) {
  y_pred <- rep(model$init_pred, nrow(X))
  eta <- model$params$eta
  
  for (tree in model$trees) {
    y_pred <- y_pred + eta * predict_one_tree(tree, X)
  }
  return(sigmoid(y_pred))
}

#  导出特征重要性
xgb_importance_manual <- function(feature_names, model) {
  raw <- model$feature_importance_raw
  
  res_list <- list()
  for (i in seq_along(feature_names)) {
    idx_str <- as.character(i)
    gain <- ifelse(is.null(raw$gain[[idx_str]]), 0, raw$gain[[idx_str]])
    cover <- ifelse(is.null(raw$cover[[idx_str]]), 0, raw$cover[[idx_str]])
    freq <- ifelse(is.null(raw$weight[[idx_str]]), 0, raw$weight[[idx_str]])
    
    if (freq > 0) {
        res_list[[i]] <- data.frame(
            Feature = feature_names[i],
            Gain = gain,
            Cover = cover,
            Frequency = freq,
            stringsAsFactors = FALSE
        )
    }
  }
  
  if (length(res_list) == 0) return(data.frame(Feature=character(), Gain=numeric(), Cover=numeric(), Frequency=numeric()))
  
  res_df <- do.call(rbind, res_list)
  
  if (sum(res_df$Gain) > 0) res_df$Gain <- res_df$Gain / sum(res_df$Gain)
  if (sum(res_df$Cover) > 0) res_df$Cover <- res_df$Cover / sum(res_df$Cover)
  if (sum(res_df$Frequency) > 0) res_df$Frequency <- res_df$Frequency / sum(res_df$Frequency)
  
  res_df <- res_df[order(res_df$Gain, decreasing = TRUE), ]
  return(res_df)
}