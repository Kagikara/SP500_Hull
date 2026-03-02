# ============================================================
# 模块 2: Logistic Regression 训练器 (glmnet版)
# 文件路径: code_section4/RF_NEW/02_model_trainer_lr.R
# ============================================================

library(glmnet)
library(dplyr)
library(caret)

train_lr_model <- function(data_bundle, config) {
  
  cat("[Model-LR] 启动 Logistic Regression (Lasso/ElasticNet) 训练...\n")
  
  # 1. 数据准备 (glmnet 需要矩阵)
  X_train_mat <- as.matrix(data_bundle$X_train)
  y_train_vec <- as.numeric(as.character(data_bundle$y_train)) # 0/1
  X_test_mat  <- as.matrix(data_bundle$X_test)
  
  # 参数提取
  lp <- config$lr_params
  # alpha: 1=Lasso (稀疏), 0=Ridge (稠密), 0.5=ElasticNet
  alpha_val <- if(is.null(lp$alpha)) 1 else lp$alpha 
  
  # ============================================================
  # 2. 内部 CV 获取 OOF 概率 (Sniper 冷启动核心)
  # ============================================================
  cat("[Model-LR] 执行 5-Fold CV 获取历史概率...\n")
  
  set.seed(config$seed)
  k_folds <- 5
  folds <- caret::createFolds(y_train_vec, k = k_folds, list = TRUE)
  
  cv_probs_oof <- numeric(length(y_train_vec))
  
  for(k in 1:k_folds) {
    val_idx <- folds[[k]]
    train_idx <- setdiff(seq_along(y_train_vec), val_idx)
    
    # 局部训练 (带 CV 寻找最佳 Lambda)
    # cv.glmnet 内部会自动再做一次 CV 来找 lambda，虽然有点套娃，但为了严谨
    cv_fit <- cv.glmnet(
      x = X_train_mat[train_idx, ], 
      y = y_train_vec[train_idx], 
      family = "binomial",
      alpha = alpha_val,
      type.measure = "auc",
      nfolds = 5
    )
    
    # 预测验证集 (使用 lambda.min 或 lambda.1se)
    # lambda.1se 会得到更稀疏的模型(防过拟合)，lambda.min 准确率略高
    preds <- predict(cv_fit, newx = X_train_mat[val_idx, ], s = "lambda.min", type = "response")
    cv_probs_oof[val_idx] <- as.numeric(preds)
  }
  
  cat(sprintf("[Model-LR] OOF AUC: %.4f\n", ModelMetrics::auc(y_train_vec, cv_probs_oof)))
  
  # ============================================================
  # 3. 概率校准 (Platt Scaling) - 保持架构一致性
  # ============================================================
  # LR 虽然输出是概率，但加上 Platt 可以修正正则化带来的偏差
  eps <- 1e-6
  cv_probs_clipped <- pmax(eps, pmin(1 - eps, cv_probs_oof))
  cv_logits <- log(cv_probs_clipped / (1 - cv_probs_clipped))
  
  platt_params <- algo_platt_fit(cv_logits, y_train_vec)
  train_cv_probs_calibrated <- algo_platt_predict(cv_logits, platt_params)
  
  # ============================================================
  # 4. 全量训练
  # ============================================================
  cat("[Model-LR] 全量训练 (寻找全局最佳 Lambda)...\n")
  
  final_cv_fit <- cv.glmnet(
    x = X_train_mat, 
    y = y_train_vec, 
    family = "binomial",
    alpha = alpha_val,
    type.measure = "auc"
  )
  
  best_lambda <- final_cv_fit$lambda.min
  cat(sprintf("  - 最佳 Lambda: %.6f\n", best_lambda))
  
  # 提取非零系数 (看看哪些因子被选中了)
  coef_list <- coef(final_cv_fit, s = "lambda.min")
  coef_df <- data.frame(
    Feature = rownames(coef_list),
    Coefficient = as.numeric(coef_list[,1])
  ) %>% 
    filter(Feature != "(Intercept)") %>%
    arrange(desc(abs(Coefficient)))
  
  # 只是为了保持接口一致，列名仍叫 Frequency (实际是 Abs Coef)
  imp_df <- coef_df %>% mutate(Frequency = abs(Coefficient)) 
  
  write.csv(imp_df, file.path(config$output_dir, "lr_coefficients.csv"), row.names = FALSE)
  
  # ============================================================
  # 5. 测试集预测
  # ============================================================
  test_probs_raw <- predict(final_cv_fit, newx = X_test_mat, s = "lambda.min", type = "response")
  test_probs_clipped <- pmax(eps, pmin(1 - eps, as.numeric(test_probs_raw)))
  test_scores_logit  <- log(test_probs_clipped / (1 - test_probs_clipped))
  
  return(list(
    model              = final_cv_fit,
    test_scores        = test_scores_logit,
    platt_params       = platt_params,
    train_cv_probs     = train_cv_probs_calibrated, # Sniper 必须
    feature_importance = imp_df,
    model_type         = "logistic_regression"
  ))
}