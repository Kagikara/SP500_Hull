# ============================================================
# 模块 2: 模型训练器
# ============================================================

library(dplyr)

# F1
calc_f1 <- function(scores, y_true) {
  y_pred <- ifelse(scores >= 0, 1, 0)
  y_true <- as.numeric(as.character(y_true))
  tp <- sum(y_pred == 1 & y_true == 1)
  fp <- sum(y_pred == 1 & y_true == 0)
  fn <- sum(y_pred == 0 & y_true == 1)
  precision <- if ((tp + fp) == 0) 0 else tp / (tp + fp)
  recall    <- if ((tp + fn) == 0) 0 else tp / (tp + fn)
  if ((precision + recall) == 0) 0 else 2 * precision * recall / (precision + recall)
}

train_model_entry <- function(data_bundle, config) {
  model_type <- if(is.null(config$model_type)) "svm" else config$model_type
  if (model_type == "svm") return(train_svm_model(data_bundle, config))
  else stop("不支持非SVM模型")
}

train_svm_model <- function(data_bundle, config) {
  
  X_train <- as.matrix(data_bundle$X_train)
  y_train <- data_bundle$y_train
  X_test  <- as.matrix(data_bundle$X_test)
  
  # 获取核函数类型，默认为 Linear
  k_type <- if(is.null(config$kernel_type)) "linear" else config$kernel_type
  cat(sprintf("[Model] 初始化 SVM 训练流程, Kernel = %s\n", toupper(k_type)))
  
  # ============================================================
  # 网格搜索
  # ============================================================
  
  if (k_type == "linear") {
    tune_grid <- expand.grid(C = config$tune_grid$C, gamma = 0) # gamma 0 占位
    cat("[Model] 线性核模式: 仅搜索参数 C\n")
  } else {
    tune_grid <- expand.grid(C = config$tune_grid$C, gamma = config$tune_grid$gamma)
    cat("[Model] RBF核模式: 搜索参数 C 和 Gamma\n")
  }
  
  best_score <- -1
  best_params <- list(C = 0.1, gamma = 0.01) # 默认
  
  k_folds <- config$cv_folds
  folds <- cut(seq(1, nrow(X_train)), breaks = k_folds, labels = FALSE)
  
  for(i in 1:nrow(tune_grid)) {
    curr_C <- tune_grid$C[i]
    curr_g <- tune_grid$gamma[i]
    
    if (k_type == "linear") {
      cat(sprintf("  > Grid Search [%d/%d] C=%.4f ... ", i, nrow(tune_grid), curr_C))
    } else {
      cat(sprintf("  > Grid Search [%d/%d] C=%.4f Gamma=%.4f ... ", i, nrow(tune_grid), curr_C, curr_g))
    }
    
    fold_scores <- numeric(k_folds)
    
    for(k in 1:k_folds) {
      idx_val <- which(folds == k)
      idx_tr  <- which(folds != k)
      
      # 训练 Fold 模型
      tm_model <- algo_svm_train_smo(
        X_train[idx_tr, ], y_train[idx_tr], 
        C = curr_C, gamma = curr_g, 
        kernel_type = k_type,
        max_iter = 5000 # 搜索时加速
      )
      
      val_s <- algo_svm_predict_score(tm_model, X_train[idx_val, ])
      fold_scores[k] <- calc_f1(val_s, y_train[idx_val])
    }
    
    avg_score <- mean(fold_scores, na.rm=TRUE)
    cat(sprintf("F1=%.4f\n", avg_score))
    
    if(avg_score > best_score) {
      best_score <- avg_score
      best_params <- list(C = curr_C, gamma = curr_g)
    }
  }
  
gamma_str <- if (!is.null(best_params$gamma)) sprintf(", Gamma=%.4f", best_params$gamma) else ""

cat(sprintf("[Model] 最佳参数找到: C=%.4f%s\n", best_params$C, gamma_str))
  
  # ============================================================
  # 2. 内部CV拟合Platt Scaling
  # ============================================================
  cat("[Model] 执行 Internal CV 以校准概率...\n")
  
  n_calib_folds <- 3
  folds_calib <- cut(seq(1, nrow(X_train)), breaks = n_calib_folds, labels = FALSE)
  cv_scores <- numeric(nrow(X_train))
  
  for(k in 1:n_calib_folds) {
    idx_val <- which(folds_calib == k)
    idx_tr  <- which(folds_calib != k)
    
    sub_model <- algo_svm_train_smo(
      X_train[idx_tr, ], y_train[idx_tr], 
      C = best_params$C, gamma = best_params$gamma,
      kernel_type = k_type,
      max_iter = 100 
    )
    cv_scores[idx_val] <- algo_svm_predict_score(sub_model, X_train[idx_val, ])
  }
  
  platt_params <- algo_platt_fit(cv_scores, as.numeric(as.character(y_train)))
  
  # ============================================================
  # 3. 训练最终模型
  # ============================================================
  cat("[Model] 使用最佳参数训练全量模型...\n")
  
  final_model <- algo_svm_train_smo(
    X_train, y_train, 
    C = best_params$C, gamma = best_params$gamma,
    kernel_type = k_type,
    max_iter = 1000 # 保证收敛
  )
  
  # 统计 SV
  n_sv <- length(final_model$sv_indices)
  sv_ratio <- n_sv / nrow(X_train)
  cat(sprintf("[Model-Stats] SV 占比: %.2f%% (Kernel: %s)\n", sv_ratio * 100, k_type))
  
  test_scores <- algo_svm_predict_score(final_model, X_test)
  saveRDS(final_model, file.path(config$output_dir, "svm_best_model.rds"))
  
  return(list(
    model = final_model,
    test_scores = test_scores,
    platt_params = platt_params,
    best_params = best_params,
    sv_ratio = sv_ratio
  ))
}