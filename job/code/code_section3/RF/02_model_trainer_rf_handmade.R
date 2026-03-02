# ============================================================
# 模块 2: RF模型训练器
# ============================================================

library(dplyr)
library(foreach)
library(doParallel)

train_rf_model <- function(data_bundle, config) {
  
  cat("[Model-RF] 启动自研随机森林训练\n")

  X_train_mat <- as.matrix(data_bundle$X_train)
  y_train_vec <- data_bundle$y_train # factor
  X_test_mat  <- as.matrix(data_bundle$X_test)
  
  # ----------------------------------------------------------------
  # 训练最终的全量模型
  # ----------------------------------------------------------------

  num_cores <- parallel::detectCores() - 1
  if (num_cores < 1) num_cores <- 1
  
  cat("  - 并行配置: 启用", num_cores, "个核心\n")
  cl <- makeCluster(num_cores)
  registerDoParallel(cl)
  
  total_trees <- config$rf_params$ntree
  n_features <- ncol(X_train_mat)
  mtry_val <- if(config$rf_params$mtry == "sqrt") floor(sqrt(n_features)) else config$rf_params$mtry
  max_depth_val <- if(!is.null(config$rf_params$max_depth)) config$rf_params$max_depth else 5
  min_node_val  <- config$rf_params$nodesize
  
  custom_functions <- c("algo_rf_train_custom", "algo_cart_build_tree", 
                        "algo_cart_find_split", "algo_cart_gini", "algo_cart_predict_batch", "algo_cart_predict_sample")
  
  cat("  - [Main] 训练全量森林 (Trees=", total_trees, ")...\n")
  
  parallel_rf_train <- function(X, y, n_trees, n_cores, mtry, depth, min_node) {
    foreach(
      i = 1:n_cores,
      .combine = c, 
      .export = custom_functions 
    ) %dopar% {
      set.seed(42 + i) # 确保随机性
      
      ntree_chunk <- floor(n_trees / n_cores)
      if (i == n_cores) ntree_chunk <- n_trees - (ntree_chunk * (n_cores - 1))
      
      algo_rf_train_custom(
        X = X, y = y,
        ntree = ntree_chunk, mtry = mtry,
        max_depth = depth, min_node_size = min_node
      )
    }
  }
  
  full_forest <- parallel_rf_train(X_train_mat, y_train_vec, total_trees, num_cores, mtry_val, max_depth_val, min_node_val)
  
  # ----------------------------------------------------------------
  # 内部K-Fold CV获取无偏Logit分数
  # ----------------------------------------------------------------
  cat("  - [Calibration] 正在执行 3-Fold CV 以拟合 Platt 参数...\n")
  
  k_folds <- 3
  folds <- cut(seq(1, nrow(X_train_mat)), breaks = k_folds, labels = FALSE)
  cv_logits <- numeric(nrow(X_train_mat))
  for(k in 1:k_folds) {
    idx_val <- which(folds == k)
    idx_train <- which(folds != k)
    
    X_tr_fold <- X_train_mat[idx_train, , drop=FALSE]
    y_tr_fold <- y_train_vec[idx_train]
    X_val_fold <- X_train_mat[idx_val, , drop=FALSE]
    cv_ntree <- max(10, floor(total_trees / 2)) 
    
    cat(sprintf("    > Fold %d/%d (Train: %d, Val: %d)...\n", k, k_folds, length(idx_train), length(idx_val)))
    
    fold_forest <- parallel_rf_train(X_tr_fold, y_tr_fold, cv_ntree, num_cores, mtry_val, max_depth_val, min_node_val)
    
    n_trees_fold <- length(fold_forest)
    all_preds <- matrix(0, nrow = nrow(X_val_fold), ncol = n_trees_fold)
    
    for (t in 1:n_trees_fold) {
      all_preds[, t] <- algo_cart_predict_batch(fold_forest[[t]], X_val_fold)
    }

    val_probs <- rowMeans(all_preds)
    
    eps <- 1e-6
    val_probs_clipped <- pmax(eps, pmin(1 - eps, val_probs))
    cv_logits[idx_val] <- log(val_probs_clipped / (1 - val_probs_clipped))
  }
  
  cat("  - [Calibration] 拟合 Sigmoid\n")
  platt_params <- algo_platt_fit(cv_logits, as.numeric(as.character(y_train_vec)))
  
  cat(sprintf("    > 参数 A: %.4f, B: %.4f\n", platt_params$A, platt_params$B))
  
  # ----------------------------------------------------------------
  # 生成测试集预测
  # ----------------------------------------------------------------
  cat("[Model-RF] 生成测试集最终预测...\n")
  
  n_test <- nrow(X_test_mat)
  n_trees_full <- length(full_forest)
  
  test_preds_mat <- matrix(0, nrow = n_test, ncol = n_trees_full)
  
  pb <- txtProgressBar(min = 0, max = n_trees_full, style = 3)
  for (i in 1:n_trees_full) {
    test_preds_mat[, i] <- algo_cart_predict_batch(full_forest[[i]], X_test_mat)
    setTxtProgressBar(pb, i)
  }
  close(pb)
  
  test_probs_raw <- rowMeans(test_preds_mat)
  
  eps <- 1e-6
  probs_clipped <- pmax(eps, pmin(1 - eps, test_probs_raw))
  test_scores_logit <- log(probs_clipped / (1 - probs_clipped))
  
  stopCluster(cl)
  
  # ----------------------------------------------------------------
  # 特征重要性
  # ----------------------------------------------------------------
  cat("[Model-RF] 统计特征重要性...\n")
  feat_counts <- table(unlist(lapply(full_forest, function(tree) {
    extract_feats <- function(node) {
      if (node$type == "leaf") return(NULL)
      c(node$feature_idx, extract_feats(node$left), extract_feats(node$right))
    }
    extract_feats(tree)
  })))
  
  feat_names <- colnames(X_train_mat)
  imp_df <- data.frame(
    Feature_Idx = as.numeric(names(feat_counts)),
    Frequency = as.numeric(feat_counts)
  )
  imp_df$Feature <- feat_names[imp_df$Feature_Idx]
  imp_df <- imp_df %>% arrange(desc(Frequency))
  
  write.csv(imp_df, file.path(config$output_dir, "rf_handmade_importance.csv"), row.names = FALSE)
  saveRDS(full_forest, file.path(config$output_dir, "rf_handmade_model.rds"))
  
  return(list(
    model = full_forest,
    test_scores = test_scores_logit,
    platt_params = platt_params,
    feature_importance = imp_df,
    model_type = "random_forest_handmade"
  ))
}