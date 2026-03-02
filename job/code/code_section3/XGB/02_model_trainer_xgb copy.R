# ============================================================
# 模块 2: XGBoost 训练器
# ============================================================
library(dplyr)


source("C:/Users/kagirasu/works_H/ml_HULL/job/code/code_section3/XGB/manual_xgboost.R") 
source("XGB/lasso.r")

train_xgb_model <- function(data_bundle, config) {
  
  cat("[Model-ManualXGB] 准备数据矩阵...\n")
  
  X_full <- as.matrix(data_bundle$X_train)
  y_full <- as.numeric(as.character(data_bundle$y_train))
  X_test <- as.matrix(data_bundle$X_test)
  
  # ============================================================
  # Lasso 特征选择
  # ============================================================
  set.seed(42)
  
  if (!is.null(config$use_lasso) && config$use_lasso == TRUE) {
    cat("[Model-ManualXGB] 启动 Lasso 特征选择...\n")
    
    lasso_res <- algo_lasso_cv(
      X_raw = X_full, 
      y_raw = y_full, 
      k_folds = config$lasso_params$k_folds, 
      n_lambdas = config$lasso_params$n_lambdas
    )
    
    kept_features <- lasso_res$selected_features
    cat(sprintf("  - Lasso 筛选: %d -> %d 个特征\n", ncol(X_full), length(kept_features)))
    
    if (length(kept_features) < 2) {
      warning("Lasso 选出特征过少，回退到全量特征。")
    } else {
      X_full <- X_full[, kept_features, drop = FALSE]
      X_test <- X_test[, kept_features, drop = FALSE]
    }
  }
  
  final_features <- colnames(X_full)
  
  params <- list(
    eta = config$xgb_params$eta,
    max_depth = config$xgb_params$max_depth,
    lambda_reg = 1.0, 
    gamma = 0.0  
  )
  
  # ============================================================
  # 2. 获取 Platt 校准参数
  # ============================================================
  cat("[Model-ManualXGB] 执行手动 5-Fold CV 以获取 OOF 预测...\n")
  
  n_folds <- 5
  folds <- sample(rep(1:n_folds, length.out = nrow(X_full)))
  oof_probs <- numeric(nrow(X_full))
  best_iters <- numeric(n_folds)
  
  for (k in 1:n_folds) {
    val_idx <- which(folds == k)
    X_tr_cv <- X_full[-val_idx, , drop = FALSE]
    y_tr_cv <- y_full[-val_idx]
    X_val_cv <- X_full[val_idx, , drop = FALSE]
    y_val_cv <- y_full[val_idx]
    
    n_rounds_cv <- config$xgb_params$nrounds
    
    cv_model <- xgb_train_manual(
      X = X_tr_cv,
      y = y_tr_cv,
      params = params,
      nrounds = n_rounds_cv,
      eval_set = list(data = X_val_cv, label = y_val_cv),
      verbose = FALSE
    )
    
    val_losses <- cv_model$history$val_loss
    best_t <- which.min(val_losses)
    best_iters[k] <- best_t

    cv_model_best <- cv_model
    cv_model_best$trees <- cv_model$trees[1:best_t]
    
    oof_probs[val_idx] <- xgb_predict_manual(cv_model_best, X_val_cv)
    
    cat(sprintf("    Fold %d: Best Iter = %d, Min Loss = %.4f\n", k, best_t, min(val_losses)))
  }
  
  avg_best_iter <- floor(mean(best_iters))
  cat(sprintf("  - CV 平均最佳轮数: %d\n", avg_best_iter))

  # Platt Fit
  eps <- 1e-6
  oof_probs_clipped <- pmax(eps, pmin(1 - eps, oof_probs))
  oof_scores <- log(oof_probs_clipped / (1 - oof_probs_clipped))
  
  real_platt_params <- algo_platt_fit(oof_scores, y_full)
  
  cat(sprintf("  - [校准完成] Platt参数: A=%.4f, B=%.4f\n", 
              real_platt_params$A, real_platt_params$B))
  
  # ============================================================
  # 全量重新训练
  # ============================================================
  cat("[Model-ManualXGB] 使用全量数据进行最终训练...\n")
  
  final_model <- xgb_train_manual(
    X = X_full,
    y = y_full,
    params = params,
    nrounds = avg_best_iter,
    verbose = FALSE
  )
  
  # ============================================================
  # 4. 生成测试集预测
  # ============================================================
  test_probs <- xgb_predict_manual(final_model, X_test)
  probs_clipped <- pmax(eps, pmin(1 - eps, test_probs))
  test_scores_logit <- log(probs_clipped / (1 - probs_clipped))
  
  # 生成特征重要性 csv
  importance_matrix <- xgb_importance_manual(feature_names = final_features, model = final_model)
  write.csv(importance_matrix, file.path(config$output_dir, "xgb_feature_importance.csv"), row.names = FALSE)
  
  # ============================================================
  # 返回结果
  # ============================================================
  return(list(
    model = final_model,
    test_scores = test_scores_logit,
    platt_params = real_platt_params,
    feature_importance = importance_matrix,
    model_type = "manual_xgboost"
  ))
}