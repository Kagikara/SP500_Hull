# ============================================================
# 模块 2: 模型训练器
# ============================================================

library(dplyr)
library(foreach)
library(doParallel)

train_rf_model <- function(data_bundle, config) {
  
  cat("[Model-RF] 启动随机森林训练...\n")
  
  X_train_mat <- as.matrix(data_bundle$X_train)
  y_train_vec <- data_bundle$y_train
  X_test_mat  <- as.matrix(data_bundle$X_test)
  
  num_cores <- parallel::detectCores() - 1
  if (num_cores < 1) num_cores <- 1
  cl <- makeCluster(num_cores)
  registerDoParallel(cl)
  
  rf_p <- config$rf_params
  n_features <- ncol(X_train_mat)
  mtry_val <- if(rf_p$mtry == "sqrt") floor(sqrt(n_features)) else rf_p$mtry
  
  custom_funcs <- c("algo_rf_train_custom", "algo_cart_build_tree", 
                    "algo_cart_find_split", "algo_cart_gini", 
                    "algo_cart_predict_batch", "algo_cart_predict_sample")
  
  run_parallel_rf <- function(X, y, n_tree_total) {
    
    base_count <- floor(n_tree_total / num_cores)
    remainder  <- n_tree_total %% num_cores
    
    tree_counts <- rep(base_count, num_cores)
    if (remainder > 0) {
      tree_counts[1:remainder] <- tree_counts[1:remainder] + 1
    }
    
    forest_chunk <- foreach(
      n_tree_local = tree_counts, 
      i = 1:num_cores,
      .combine = c, 
      .export = c(custom_funcs, "rf_p", "mtry_val") 
    ) %dopar% {
      
      if(n_tree_local <= 0) return(NULL)
      
      set.seed(42 + i)
      
      algo_rf_train_custom(
        X = X, y = y,
        ntree = n_tree_local,
        mtry = mtry_val,
        max_depth = rf_p$max_depth,
        min_node_size = rf_p$nodesize
      )
    }
    return(forest_chunk)
  }
  
  # ============================================================
  # 3. 内部 CV 获取无偏 Logits
  # ============================================================
  cat("[Model-RF] 执行 3-Fold CV 以校准概率\n")
  
  k_folds <- 3
  folds <- cut(seq(1, nrow(X_train_mat)), breaks = k_folds, labels = FALSE)
  cv_logits <- numeric(nrow(X_train_mat))
  
  cv_ntree_total <- max(50, floor(rf_p$ntree / 5)) 
  
  for(k in 1:k_folds) {
    idx_val <- which(folds == k)
    idx_tr  <- which(folds != k)

    sub_forest <- run_parallel_rf(X_train_mat[idx_tr, ], y_train_vec[idx_tr], cv_ntree_total)

    n_trees_sub <- length(sub_forest)
    if(n_trees_sub > 0) {
      preds_mat <- matrix(0, nrow = length(idx_val), ncol = n_trees_sub)
      X_val_sub <- X_train_mat[idx_val, , drop=FALSE]
      
      for(t in 1:n_trees_sub) {
        preds_mat[, t] <- algo_cart_predict_batch(sub_forest[[t]], X_val_sub)
      }
      val_probs <- rowMeans(preds_mat)
    } else {
      val_probs <- rep(0.5, length(idx_val))
    }

    eps <- 1e-6
    val_probs_clip <- pmax(eps, pmin(1 - eps, val_probs))
    cv_logits[idx_val] <- log(val_probs_clip / (1 - val_probs_clip))
    
    cat(sprintf("  - Fold %d/%d 完成.\n", k, k_folds))
  }

  platt_params <- algo_platt_fit(cv_logits, as.numeric(as.character(y_train_vec)))
  cat(sprintf("  - 校准参数: A=%.4f, B=%.4f\n", platt_params$A, platt_params$B))

  train_cv_probs <- algo_platt_predict(cv_logits, platt_params)

  # ============================================================
  # 4. 训练全量模型
  # ============================================================
  cat("[Model-RF] 训练全量模型 (Trees=", rf_p$ntree, ")...\n")
  t_start <- Sys.time()
  
  full_forest <- run_parallel_rf(X_train_mat, y_train_vec, rf_p$ntree)
  
  t_end <- Sys.time()
  cat("  - 训练耗时:", round(difftime(t_end, t_start, units="secs"), 2), "秒\n")
  
  # 停止集群
  stopCluster(cl)
  
  # ============================================================
  # 5. 预测测试集
  # ============================================================
  cat("[Model-RF] 生成测试集预测...\n")
  
  n_test <- nrow(X_test_mat)
  n_trees_full <- length(full_forest)
  test_preds_mat <- matrix(0, nrow = n_test, ncol = n_trees_full)
  
  for (i in 1:n_trees_full) {
    test_preds_mat[, i] <- algo_cart_predict_batch(full_forest[[i]], X_test_mat)
  }
  test_probs_raw <- rowMeans(test_preds_mat)
  
  eps <- 1e-6
  probs_clipped <- pmax(eps, pmin(1 - eps, test_probs_raw))
  test_scores_logit <- log(probs_clipped / (1 - probs_clipped))

  cat("[Model-RF] 统计特征重要性...\n")
  feat_counts <- table(unlist(lapply(full_forest, function(tree) {
    extract_feats <- function(node) {
      if (node$type == "leaf") return(NULL)
      c(node$feature_idx, extract_feats(node$left), extract_feats(node$right))
    }
    extract_feats(tree)
  })))
  
  imp_df <- data.frame(
    Feature_Idx = as.numeric(names(feat_counts)),
    Frequency = as.numeric(feat_counts)
  )
  imp_df$Feature <- colnames(X_train_mat)[imp_df$Feature_Idx]
  imp_df <- imp_df %>% arrange(desc(Frequency))
  
  write.csv(imp_df, file.path(config$output_dir, "rf_handmade_importance.csv"), row.names = FALSE)
  saveRDS(full_forest, file.path(config$output_dir, "rf_handmade_model.rds"))
  
  return(list(
    model = full_forest,
    test_scores = test_scores_logit,
    platt_params = platt_params, 
    train_cv_probs = train_cv_probs,
    feature_importance = imp_df,
    model_type = "random_forest_handmade"
  ))
}